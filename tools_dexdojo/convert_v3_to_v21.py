# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Convert DexDojo LeRobot v3.0 buckets into a v2.1-style tree for lerobot 0.3.3.

v3 layout (source):
    meta/info.json
    meta/episodes/chunk-XXX/file-XXX.parquet   (multi-episode metadata)
    meta/tasks.parquet
    data/chunk-XXX/file-XXX.parquet            (multi-episode frames)
    videos/<cam>/chunk-XXX/file-XXX.mp4        (multi-episode concatenated)

v2.1 layout (output, what LatentLeRobotDataset needs):
    meta/info.json          (codebase_version v2.1, per-episode data_path)
    meta/episodes.jsonl     (+ custom `action_config` single segment 0..length)
    meta/tasks.jsonl
    data/chunk-XXX/episode_XXXXXX.parquet     (per-episode slices)
    latents/                (filled later by extract_latents.py)

Videos are NOT copied: training reads only latents + parquet actions, and
extract_latents.py decodes the v3 concatenated files directly. A video map
JSON (episode -> source file + frame range) is emitted for the extractor.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def convert_task(src: Path, dst: Path, task_text_override=None):
    info = json.loads((src / "meta" / "info.json").read_text())
    assert info["codebase_version"].startswith("v3"), info["codebase_version"]

    eps_table = pq.read_table(
        next((src / "meta" / "episodes").glob("chunk-*/*.parquet")))
    eps = eps_table.to_pylist()
    eps.sort(key=lambda e: e["episode_index"])

    # task strings: one per episode (identical across episodes in DexDojo)
    tasks = eps[0]["tasks"]
    task_text = task_text_override or tasks[0]

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ---- per-episode parquet slices --------------------------------------
    data_files = {}  # (chunk_index, file_index) -> parquet path cache
    episode_rows = []
    episodes_stats_rows = []
    video_map = []   # for extract_latents.py
    cursor = 0
    for e in eps:
        ei = e["episode_index"]
        length = e["length"]
        dkey = (e["data/chunk_index"], e["data/file_index"])
        if dkey not in data_files:
            data_files[dkey] = pq.read_table(
                src / f"data/chunk-{dkey[0]:03d}/file-{dkey[1]:03d}.parquet")
        table = data_files[dkey]
        frm, to = e["dataset_from_index"], e["dataset_to_index"]
        assert to - frm == length, (ei, frm, to, length)
        out_parquet = dst / f"data/chunk-000/episode_{ei:06d}.parquet"
        sub = table.slice(frm, length)
        pq.write_table(sub, out_parquet)

        # real per-episode action stats for meta/episodes_stats.jsonl
        # (v2.1 loader requires the file; training itself never reads stats)
        act = np.stack(sub["action"].to_pylist()).astype(np.float64)
        q = np.quantile(act, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)
        episodes_stats_rows.append({
            "episode_index": ei,
            "stats": {
                "action": {
                    "min": act.min(axis=0).tolist(),
                    "max": act.max(axis=0).tolist(),
                    "mean": act.mean(axis=0).tolist(),
                    "std": act.std(axis=0).tolist(),
                    "count": [int(act.shape[0])] * act.shape[1],
                    "q01": q[0].tolist(), "q25": q[1].tolist(),
                    "q50": q[2].tolist(), "q75": q[3].tolist(),
                    "q99": q[4].tolist(),
                }
            },
        })

        episode_rows.append({
            "episode_index": ei,
            "tasks": tasks,
            "length": length,
            "dataset_from": cursor,
            "dataset_to": cursor + length,
            "action_config": [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": task_text,
            }],
        })
        video_map.append({
            "episode_index": ei,
            "length": length,
            "videos": {
                cam: {
                    "chunk_index": e[f"videos/{cam}/chunk_index"],
                    "file_index": e[f"videos/{cam}/file_index"],
                    "from_timestamp": e[f"videos/{cam}/from_timestamp"],
                    "to_timestamp": e[f"videos/{cam}/to_timestamp"],
                }
                for cam in info["features"]
                if info["features"][cam]["dtype"] == "video"
            },
        })
        cursor += length

    # ---- meta files -------------------------------------------------------
    features = json.loads(json.dumps(info["features"]))  # deep copy
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for row in episode_rows:
            f.write(json.dumps(row) + "\n")
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_text}) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in episodes_stats_rows:
            f.write(json.dumps(row) + "\n")

    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    # lerobot 0.3.3 checks that every episode's video file exists
    # (get_episodes_file_paths) even though training only reads latents and
    # the parquet action column; satisfy the check with empty placeholders.
    for k in video_keys:
        cam_dir = dst / "videos" / "chunk-000" / k
        cam_dir.mkdir(parents=True, exist_ok=True)
        for e in eps:
            dummy = cam_dir / f"episode_{e['episode_index']:06d}.mp4"
            if not dummy.exists():
                dummy.touch()

    v21_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", ""),
        "total_episodes": len(eps),
        "total_frames": cursor,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": info["fps"],
        "splits": {"train": f"0:{len(eps)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (dst / "meta" / "info.json").write_text(json.dumps(v21_info, indent=4))
    (dst / "video_map.json").write_text(json.dumps({
        "source_root": str(src),
        "fps": info["fps"],
        "episodes": video_map,
    }, indent=1))
    print(f"[convert] {src.name}: {len(eps)} episodes, {cursor} frames -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="v3 root containing the 6 task buckets")
    ap.add_argument("--dst", required=True, help="output v2.1 root")
    args = ap.parse_args()
    src_root, dst_root = Path(args.src), Path(args.dst)
    for task_dir in sorted(src_root.iterdir()):
        if not (task_dir / "meta" / "info.json").is_file():
            print(f"[skip] {task_dir}")
            continue
        convert_task(task_dir, dst_root / task_dir.name)


if __name__ == "__main__":
    main()
