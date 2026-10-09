# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Offline VAE latent extraction for DexDojo buckets (LingBot-VA format).

Reads the v2.1-converted task dirs (produced by convert_v3_to_v21.py, which
emits video_map.json), decodes the three used cameras from the concatenated
v3 mp4 files, samples every `--stride`-th frame (25 Hz -> 6.25 fps at
stride 4), and encodes each episode with the Wan2.2 causal VAE exactly the
way `wan_va_server.py::_encode_obs` does at inference:

    frames [T,H,W,3] uint8 -> permute [3,T,H,W] -> bilinear resize
    -> /255*2-1 -> bf16 -> streaming_vae.encode_chunk (cams as batch)
    -> mu (not sample) -> (mu - latents_mean) * (1/latents_std)
    -> per-camera [f*h*w, c] flattened .pth with README metadata fields

Sampling count is truncated to 1+4k video frames so the causal VAE maps it
to exactly 1+k latent frames (the training code's
`latent_frame_num = (len(frame_ids)-1)//4 + 1` assumes this).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wan_va.modules.utils import load_vae, load_text_encoder, load_tokenizer, WanVAEStreamingWrapper  # noqa: E402


def sample_count(length: int, stride: int) -> int:
    n = (length + stride - 1) // stride  # frame_ids 0, s, 2s, ...
    n = min(n, length)
    k = (n - 1) // 4
    return 1 + 4 * k


def decode_task_cam(src_root: Path, cam: str, episodes: list, stride: int,
                    height: int, width: int, out: dict):
    """Sequentially decode one camera's concatenated files; slice episodes."""
    files = {}
    for ep in episodes:
        v = ep["videos"][cam]
        files.setdefault((v["chunk_index"], v["file_index"]), []).append(ep)

    for (chunk_idx, file_idx), eps_in_file in sorted(files.items()):
        path = src_root / "videos" / cam / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        container = av.open(str(path))
        stream = container.streams.video[0]
        eps_sorted = sorted(
            eps_in_file, key=lambda e: e["videos"][cam]["from_timestamp"])
        # frame ranges from timestamps; validated against cumulative length
        bounds = []
        for e in eps_sorted:
            v = e["videos"][cam]
            start = int(round(v["from_timestamp"] * stream.average_rate))
            bounds.append((start, start + e["length"], e["episode_index"]))
        for i in range(len(bounds) - 1):
            assert bounds[i][1] <= bounds[i + 1][0] + 1, (cam, bounds[i], bounds[i + 1])

        cur = 0  # which episode we are filling
        n_read = 0
        target_count = {e["episode_index"]: sample_count(e["length"], stride)
                        for e in eps_sorted}
        frame_pos = {e["episode_index"]: [] for e in eps_sorted}
        for frame in container.decode(video=0):
            while cur < len(bounds) and n_read >= bounds[cur][1]:
                cur += 1
            if cur >= len(bounds):
                break
            ei = bounds[cur][2]
            local = n_read - bounds[cur][0]
            n_read += 1
            if local % stride:
                continue
            if len(frame_pos[ei]) >= target_count[ei]:
                continue
            img = frame.to_ndarray(format="rgb24")  # H,W,3 uint8
            out.setdefault(ei, []).append(img)
            frame_pos[ei].append(local)
        container.close()
        for e in eps_sorted:
            ei = e["episode_index"]
            assert len(out.get(ei, [])) == target_count[ei], \
                (cam, path, ei, len(out.get(ei, [])), target_count[ei])


@torch.no_grad()
def encode_episode(vae_wrapper, vae, frames_per_cam: dict, cam_keys: list,
                   height: int, width: int, device, dtype):
    """frames_per_cam: {cam: [T,H,W,3] uint8} -> {cam: latent [f*h*w, c]}"""
    videos = []
    for cam in cam_keys:
        arr = np.stack(frames_per_cam[cam])  # T,H,W,3
        t = torch.from_numpy(arr).float().permute(3, 0, 1, 2)  # 3,T,H,W
        t = F.interpolate(t.unsqueeze(0), size=(height, width),
                          mode="bilinear", align_corners=False)
        videos.append(t)
    videos = torch.cat(videos, dim=0).to(device).to(dtype) / 255.0 * 2.0 - 1.0
    enc = vae_wrapper.encode_chunk(videos)
    mu, _ = torch.chunk(enc, 2, dim=1)
    latents_mean = torch.tensor(vae.config.latents_mean).to(mu.device)
    latents_std = torch.tensor(vae.config.latents_std).to(mu.device)
    mu_norm = (mu.float() - latents_mean.view(1, -1, 1, 1, 1)) * (
        1.0 / latents_std).view(1, -1, 1, 1, 1)
    out = {}
    for i, cam in enumerate(cam_keys):
        latent = mu_norm[i].to(dtype)  # C,f,h,w
        c, f, h, w = latent.shape
        out[cam] = latent.permute(1, 2, 3, 0).reshape(-1, c)  # (f h w) c
        out[cam + "_shape"] = (f, h, w)
    return out


@torch.no_grad()
def encode_text(tokenizer, text_encoder, text, device, dtype,
                max_sequence_length=512):
    from ftfy import fix_text
    prompt = fix_text(text)
    inputs = tokenizer([prompt], padding="max_length",
                       max_length=max_sequence_length, truncation=True,
                       add_special_tokens=True, return_attention_mask=True,
                       return_tensors="pt")
    ids, mask = inputs.input_ids, inputs.attention_mask
    seq_len = mask.gt(0).sum(dim=1).long()
    emb = text_encoder(ids.to(text_encoder.device),
                       mask.to(text_encoder.device)).last_hidden_state
    emb = emb.to(dtype=dtype, device=device)
    emb = emb[0][:seq_len[0]]
    emb = torch.cat([emb, emb.new_zeros(max_sequence_length - emb.size(0),
                                        emb.size(1))])
    return emb  # [512, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="v2.1-converted task dirs root")
    ap.add_argument("--model-path", required=True,
                    help="lingbot-va base dir (vae/tokenizer/text_encoder)")
    ap.add_argument("--cam-keys", nargs="+", default=[
        "observation.images.stereo",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch-episodes", type=int, default=4)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--empty-emb-out", default=None,
                    help="also write empty text embedding here")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    vae = load_vae(str(Path(args.model_path) / "vae"), dtype, device)
    wrapper = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(str(Path(args.model_path) / "tokenizer"))
    text_encoder = load_text_encoder(
        str(Path(args.model_path) / "text_encoder"), dtype, device)
    text_encoder.eval()

    for task_dir in sorted(Path(args.data_root).iterdir()):
        if not (task_dir / "video_map.json").is_file():
            continue
        name = task_dir.name
        if args.tasks and name not in args.tasks:
            continue
        vm = json.loads((task_dir / "video_map.json").read_text())
        src_root = Path(vm["source_root"])
        episodes = vm["episodes"]
        eps_meta = {json.loads(l)["episode_index"]: json.loads(l)
                    for l in (task_dir / "meta" / "episodes.jsonl").open()}
        task_text = eps_meta[0]["action_config"][0]["action_text"]

        text_emb = encode_text(tokenizer, text_encoder, task_text,
                               device, dtype).cpu()

        # decode all cams for this task
        frames = {ei: {} for ei in eps_meta}  # ei -> {cam: [frames]}
        for cam in args.cam_keys:
            buf = {}
            decode_task_cam(src_root, cam, episodes, args.stride,
                            args.height, args.width, buf)
            for ei, imgs in buf.items():
                frames[ei][cam] = imgs

        lat_dir = task_dir / "latents" / "chunk-000"
        done = 0
        eis = sorted(frames)
        for i in range(0, len(eis), args.batch_episodes):
            batch_eis = eis[i:i + args.batch_episodes]
            wrapper.clear_cache()
            # encode one episode at a time (streaming cache is per-sequence);
            # batching across episodes would interleave cache states.
            for ei in batch_eis:
                wrapper.clear_cache()
                if not all(c in frames[ei] for c in args.cam_keys):
                    raise RuntimeError(f"{name} ep{ei}: missing cameras")
                res = encode_episode(wrapper, vae, frames[ei],
                                     args.cam_keys, args.height,
                                     args.width, device, dtype)
                meta = eps_meta[ei]
                length = meta["length"]
                n_frames = sample_count(length, args.stride)
                frame_ids = list(range(0, n_frames * args.stride, args.stride))[:n_frames]
                for cam in args.cam_keys:
                    f, h, w = res[cam + "_shape"]
                    cam_dir = lat_dir / cam
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "latent": res[cam].cpu(),
                        "latent_num_frames": f,
                        "latent_height": h,
                        "latent_width": w,
                        "video_num_frames": n_frames,
                        "video_height": args.height,
                        "video_width": args.width,
                        "text_emb": text_emb,
                        "text": task_text,
                        "frame_ids": frame_ids,
                        "start_frame": 0,
                        "end_frame": length,
                        "fps": vm["fps"] / args.stride,
                        "ori_fps": vm["fps"],
                    }
                    torch.save(payload, cam_dir /
                               f"episode_{ei:06d}_0_{length}.pth")
                done += 1
        print(f"[latents] {name}: {done}/{len(eis)} episodes")

    if args.empty_emb_out:
        empty = encode_text(tokenizer, text_encoder, "", device, dtype)
        torch.save(empty.cpu(), args.empty_emb_out)
        print(f"[latents] empty_emb -> {args.empty_emb_out}")


if __name__ == "__main__":
    main()
