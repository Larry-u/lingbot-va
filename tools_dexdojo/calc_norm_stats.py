# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Compute q01/q99 norm stats over the 54-D actions of all DexDojo buckets.

Reads the v2.1-converted per-episode parquet files (action column only),
pools all frames across tasks, and writes:

  <out>/norm_stats.json            {"q01": [...54], "q99": [...54]}
  <out>/norm_stats_literal.py      ready-to-paste `norm_stat = {...}` block
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="v2.1-converted task dirs root")
    ap.add_argument("--out", required=True, help="output dir for stats files")
    args = ap.parse_args()

    chunks = []
    for task_dir in sorted(Path(args.data_root).iterdir()):
        pq_dir = task_dir / "data" / "chunk-000"
        if not pq_dir.is_dir():
            continue
        n = 0
        for f in sorted(pq_dir.glob("episode_*.parquet")):
            arr = pq.read_table(f, columns=["action"])["action"].to_numpy()
            chunks.append(np.stack(arr).astype(np.float64))
            n += 1
        print(f"[stats] {task_dir.name}: {n} episodes")
    all_actions = np.concatenate(chunks, axis=0)
    assert all_actions.shape[1] == 54, all_actions.shape
    q01 = np.quantile(all_actions, 0.01, axis=0)
    q99 = np.quantile(all_actions, 0.99, axis=0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "norm_stats.json").write_text(json.dumps(
        {"q01": q01.tolist(), "q99": q99.tolist()}, indent=1))
    with (out / "norm_stats_literal.py").open("w") as f:
        f.write("norm_stat = {\n")
        f.write(f'    "q01": {q01.tolist()},\n')
        f.write(f'    "q99": {q99.tolist()},\n')
        f.write("}\n")
    span = q99 - q01
    print(f"[stats] frames={all_actions.shape[0]} dims={all_actions.shape[1]}")
    print(f"[stats] min span={span.min():.6f} @ dim {int(span.argmin())}, "
          f"max span={span.max():.6f}")


if __name__ == "__main__":
    main()
