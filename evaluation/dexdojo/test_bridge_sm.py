"""Standalone state-machine test for LingbotRemoteModel (no WS server).

Drives the model exactly like the DexDojo eval_one_episode driver:
reset -> [update_obs, get_action -> R rows -> R update_obs pushes]... and
asserts the lingbot-side message sequence matches the robotwin contract:
infer(reset) once per episode, first chunk trimmed, compute_kv_cache with
4/8 keyframes + full chunk between chunks, plain infer otherwise.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lingbot_policy_model as m

CALLS = []


class FakeClient:
    def infer(self, payload):
        CALLS.append(payload)
        if payload.get("reset"):
            return {}
        if payload.get("compute_kv_cache"):
            kfs = payload["obs"]
            assert isinstance(kfs, list), "kv obs must be a keyframe list"
            for kf in kfs:
                assert set(kf) == set(m.CAM_KEY_MAP[k] for k in
                                      ("cam_head", "cam_left_wrist", "cam_right_wrist")), kf.keys()
            st = payload["state"]
            assert st.shape == (54, 2, 16), st.shape
            return {}
        # plain infer: return [54, F, H]
        return {"action": np.random.randn(54, 2, 16).astype(np.float32)}


def obs(step, env_idx=0, keyframes=None):
    o = {
        "env_idx": env_idx,
        "instruction": "test task",
        "vision": {
            "cam_head": np.zeros((480, 640, 3), np.uint8),
            "cam_left_wrist": np.zeros((480, 640, 3), np.uint8),
            "cam_right_wrist": np.zeros((480, 640, 3), np.uint8),
        },
        "layout_id": 3,
        "action_index": step,
    }
    if keyframes is not None:
        o["lingbot_keyframes"] = keyframes
    return o


def run_episode(model, chunks):
    """Mirror the patched vitra_w0 driver: keyframes attach to the
    chunk-boundary obs; per-step update_obs never reach the model."""
    model.reset()
    step = 0
    pending_kfs = None
    for c in range(chunks):
        model.update_obs(obs(step, keyframes=pending_kfs))
        rows = model.get_action()
        R = len(rows)
        assert all(isinstance(r, dict) and sum(len(v) for v in r.values()) == 54
                   for r in rows)
        kfs = []
        for r in range(R):
            step += 1
            if (r + 1) % 4 == 0:
                kfs.append(obs(step))
        pending_kfs = kfs or None
    return step


def main():
    model = m.LingbotRemoteModel("ws://127.0.0.1:1")
    model._connect = lambda: None
    model._client = FakeClient()

    steps = run_episode(model, chunks=3)
    print(f"episode done: {steps} steps executed")

    seq = []
    for p in CALLS:
        if p.get("reset"):
            seq.append("R")
        elif p.get("compute_kv_cache"):
            seq.append(f"kv{len(p['obs'])}")
        else:
            seq.append("i")
    print("call sequence:", " ".join(seq))
    # expected: R i (kv4 i) (kv8 i)  — first chunk trimmed to 16 rows
    assert seq[0] == "R" and seq[1] == "i", seq
    assert seq[2] == "kv4", seq
    assert seq[3] == "i", seq
    assert seq[4] == "kv8", seq
    assert seq[5] == "i", seq
    print("STATE MACHINE OK")


if __name__ == "__main__":
    main()
