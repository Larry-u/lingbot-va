# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""DexDojo -> LingBot-VA policy adapter (the bridge model).

Duck-typed XPolicyLab model served through the DexDojo checkout's
``PolicyServer`` (same surface as the W0 bridge's ``W0RemoteModel``):
``reset`` / ``update_obs(_batch)`` / ``get_action(_batch)``. Every inference
is forwarded to a LingBot-VA websocket server
(``wan_va/wan_va_server.py --config-name dexdojo``); this process never
loads model weights.

Mapping contract (locked to the xspark_dexbench corpus):

* actions are 54-D absolute joint-position targets in interleaved block
  order ``[left_arm(7), left_hand(20), right_arm(7), right_hand(20)]`` —
  exactly what the simulator's joint-mode ``take_action`` consumes;
* cameras: the sim's ``cam_head`` is the corpus ``stereo`` view;
  wrist names match; frames are true RGB;
* the model takes no proprio input — obs state is ignored.

Chunk discipline mirrors ``evaluation/robotwin/eval_polict_client_openpi.py``
so eval stays aligned with training data slicing:

* episode start: one WS ``reset(prompt)``, then ``infer(obs=initial obs)``
  -> chunk ``[54, F, H]``; the FIRST chunk's latent-frame-0 rows (H rows of
  zero-padded pre-history in training) are NOT executed;
* per latent frame, H rows are executed and the observation after every
  4th row is a keyframe (4 actions per sampled video frame @ 25 Hz sim);
* next chunk: WS ``compute_kv_cache(obs=keyframes, state=previous chunk)``
  then ``infer``; real observations enter the AR context only here.

``update_obs`` observations are counted since the last ``get_action``: the
obs pushed after row r is push number r+1, so pushes 4, 8, 12, ... are
keyframes. The driver pushes R observations per R-row chunk (R-1 mid-chunk
plus one at the loop top) so the final keyframe of a chunk arrives as the
"pre-chunk" observation of the next ``get_action`` and is still counted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from websocket_client_policy import WebsocketClientPolicy  # noqa: E402

ARM_DIM = 7
HAND_DIM = 20
STATE_DIM = 54
# action/obs blocks in corpus order — identical to the W0 bridge contract
STATE_KEYS = (
    ("left_arm_joint_state", ARM_DIM),
    ("left_ee_joint_state", HAND_DIM),
    ("right_arm_joint_state", ARM_DIM),
    ("right_ee_joint_state", HAND_DIM),
)
# DexDojo live vision names -> lingbot-va obs_cam_keys (config order)
CAM_KEY_MAP = {
    "cam_head": "observation.images.stereo",
    "cam_high": "observation.images.stereo",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
HEAD_CAMERAS = ("cam_head", "cam_high", "head_camera", "top_camera")
LEFT_WRIST_CAMERAS = ("cam_left_wrist", "left_wrist_camera", "left_camera")
RIGHT_WRIST_CAMERAS = ("cam_right_wrist", "right_wrist_camera", "right_camera")
ACTIONS_PER_VIDEO_FRAME = 4


def split_action(row) -> dict:
    arr = np.asarray(row, dtype=np.float32).reshape(-1)
    if arr.shape[0] != STATE_DIM:
        raise ValueError(f"action row has dim {arr.shape[0]}, expected {STATE_DIM}")
    out, offset = {}, 0
    for key, dim in STATE_KEYS:
        out[key] = arr[offset:offset + dim].copy()
        offset += dim
    return out


def extract_image(vision: dict, candidates):
    for name in candidates:
        if name not in vision:
            continue
        entry = vision[name]
        if isinstance(entry, dict):
            for key in ("color", "rgb"):
                if entry.get(key) is not None:
                    return np.asarray(entry[key])
        elif entry is not None:
            return np.asarray(entry)
    raise KeyError(f"no camera among {candidates} in obs['vision'] (has: {sorted(vision)})")


def to_rgb_u8(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"camera image must be HxWx(3|4), got {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def prompt_of(obs: dict) -> str:
    prompt = obs.get("instruction") or obs.get("prompt") or ""
    if isinstance(prompt, (list, tuple)):
        prompt = prompt[0] if prompt else ""
    return str(prompt)


def format_obs(obs: dict) -> dict:
    """DexDojo obs -> lingbot-va obs dict keyed by obs_cam_keys."""
    vision = obs.get("vision") or {}
    return {
        CAM_KEY_MAP[name]: to_rgb_u8(extract_image(vision, (name,)))
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
    }


class _EnvState:
    __slots__ = ("initialized", "latest_obs", "first_obs", "full_chunk",
                 "obs_since_chunk", "keyframes", "prompt")

    def __init__(self):
        self.initialized = False
        self.latest_obs = None
        self.first_obs = None
        self.full_chunk = None      # np [54, F, H] as returned by the server
        self.obs_since_chunk = 0    # update_obs pushes since last get_action
        self.keyframes = []         # lingbot-format obs at 4-action cadence
        self.prompt = ""


class LingbotRemoteModel:
    """XPolicyLab-facing model served by ``lingbot_bridge.py``."""

    def __init__(self, server_url: str, timeout: float = 300.0):
        if not server_url:
            raise ValueError("server_url is required")
        from urllib.parse import urlparse
        parsed = urlparse(server_url if "://" in server_url
                          else f"ws://{server_url}")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._timeout = timeout
        self._client = None
        self._envs: dict[int, _EnvState] = {}
        self._last_single_env = 0

    # ----- server readiness --------------------------------------------------

    def wait_server(self, timeout_s: float = 900.0, poll_s: float = 5.0):
        """Block until the LingBot-VA server's /healthz answers."""
        import urllib.request
        deadline = time.monotonic() + float(timeout_s)
        last_err = None
        url = f"http://{self._host}:{self._port}/healthz"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        self._connect()
                        return {"policy": "lingbot-va", "server": url}
            except Exception as exc:  # noqa: BLE001 — loading/refused/reset
                last_err = exc
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"LingBot-VA server not ready after {timeout_s:.0f}s "
                        f"({url}): {last_err}"
                    ) from exc
                time.sleep(poll_s)

    def _connect(self):
        if self._client is None:
            self._client = WebsocketClientPolicy(
                host=self._host, port=self._port)

    # ----- XPolicyLab model surface -------------------------------------------

    def reset(self):
        self._envs.clear()

    def update_obs(self, obs):
        self._absorb(obs)

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self._absorb(obs)

    def get_action(self):
        env = self._envs.get(self._last_single_env)
        if env is None or env.latest_obs is None:
            raise RuntimeError("update_obs must be called before get_action")
        return [split_action(row) for row in self._chunk_rows_env(env)]

    def get_action_batch(self, env_idx_list=None):
        if not self._envs:
            raise RuntimeError("update_obs_batch must be called before get_action_batch")
        if env_idx_list is not None:
            envs = [self._envs[int(i)] for i in env_idx_list]
        else:
            envs = list(self._envs.values())
        return [[split_action(row) for row in self._chunk_rows_env(env)]
                for env in envs]

    # ----- chunk pipeline -------------------------------------------------------

    def _absorb(self, obs):
        env_idx = int(obs.get("env_idx", 0))
        self._last_single_env = env_idx
        env = self._envs.get(env_idx)
        if env is None:
            env = self._envs[env_idx] = _EnvState()
        # the patched vitra_w0 driver attaches the 4-step-cadence keyframes
        # to the chunk-boundary obs: the stock model_client swallows per-step
        # update_obs client-side, so this is the only channel they can
        # arrive on. The kv state stays env.full_chunk (the FULL server
        # chunk incl. the skipped frame-0 rows; the driver's echo would be
        # the trimmed rows only).
        obs.pop("lingbot_prev_chunk", None)
        kfs = obs.pop("lingbot_keyframes", None)
        if kfs:
            env.keyframes.extend(format_obs(k) for k in kfs)
        env.latest_obs = obs
        if env.full_chunk is None:
            return  # pre-first-chunk obs (also the first_obs source)
        env.obs_since_chunk += 1

    def _chunk_rows_env(self, env: _EnvState):
        self._connect()
        print(f"[lingbot-bridge] get_action: initialized={env.initialized} "
              f"obs_since_chunk={env.obs_since_chunk} "
              f"keyframes={len(env.keyframes)} "
              f"frame_st_chunk_shape="
              f"{None if env.full_chunk is None else env.full_chunk.shape}",
              flush=True)
        if not env.initialized:
            env.initialized = True
            env.prompt = prompt_of(env.latest_obs)
            env.first_obs = format_obs(env.latest_obs)
            self._client.infer(dict(reset=True, prompt=env.prompt))
            env.full_chunk = self._infer_chunk(env)
            rows = self._rows_to_execute(env, first=True)
        else:
            if env.keyframes:
                self._client.infer(dict(
                    obs=list(env.keyframes), compute_kv_cache=True,
                    state=env.full_chunk,
                ))
                env.keyframes = []
            # else: driver re-asks without new keyframes (episode-end edge)
            env.full_chunk = self._infer_chunk(env)
            rows = self._rows_to_execute(env, first=False)
        env.obs_since_chunk = 0
        return rows

    def _infer_chunk(self, env: _EnvState) -> np.ndarray:
        resp = self._client.infer(dict(obs=env.first_obs, prompt=env.prompt))
        action = np.asarray(resp["action"], dtype=np.float32)
        if action.shape[0] != STATE_DIM:
            raise RuntimeError(
                f"server returned {action.shape[0]}-D actions, expected {STATE_DIM}")
        return action  # [54, F, H]

    @staticmethod
    def _rows_to_execute(env: _EnvState, first: bool):
        # first chunk skips latent frame 0 (H zero-padded pre-history rows)
        chunk = env.full_chunk[:, 1:, :] if first else env.full_chunk
        return [chunk[:, i, j]
                for i in range(chunk.shape[1])
                for j in range(chunk.shape[2])]
