#!/usr/bin/env python3
"""Aggregate DexDojo eval results for one run dir into summary.tsv.

Reads every ``view/eval_result/DexBench/<task>/vitra_w0/*/*/*/_result.json``
(one per (task, timestamp) invocation; layout details are merged by
layout_id, latest timestamp wins) and writes ``summary.tsv`` with one row
per (task, seed) plus per-seed and grand TOTAL rows, mirroring the shared
service's multi-seed schema.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

TASKS = ["collect_objects", "dual_bottles_pick", "hammer_beat",
         "insert_block", "retrieve_gap", "stack_bowls"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--num", type=int, default=50, help="episodes per item")
    args = ap.parse_args()

    run = Path(args.run)
    # item dirs we launched carry logs named sim_<task>__s<seed>.log; the
    # sim itself nests by task only, so attribute layouts to seeds by the
    # recorded per-item logs' timestamp mapping is unreliable — instead we
    # require the caller to have used one seed per view pass OR infer from
    # eval_policy's additional_info. Simplest robust scheme: the runner
    # stamps each item's layouts via --additional_info "run|task|seed".
    root = run / "view" / "eval_result" / "DexBench"
    per_item = defaultdict(dict)  # (task, seed) -> {layout_id: (success, score)}
    if not root.is_dir():
        print(f"[summarize] no eval_result under {run}")
        return

    for res in root.glob("*/*/*/*/*/_result.json"):
        parts = res.parts
        task = parts[parts.index("DexBench") + 1]
        run_token = parts[-4]  # <additional_info> dir level: run__task__s<seed>
        seed = 0
        bits = run_token.rsplit("__", 2)
        if len(bits) == 3 and bits[2].startswith("s") and bits[2][1:].isdigit():
            seed = int(bits[2][1:])
        try:
            data = json.loads(res.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for det in data.get("details", {}).values():
            lid = int(det.get("layout_id", -1))
            per_item[(task, seed)][lid] = (bool(det.get("success", False)),
                                           float(det.get("score", 0.0)))

    rows = []
    totals = defaultdict(lambda: [0, 0, 0.0, 0])  # seed -> [succ, n, score, n]
    for (task, seed) in sorted(per_item, key=lambda x: (TASKS.index(x[0]) if x[0] in TASKS else 99, x[1])):
        d = per_item[(task, seed)]
        expected = list(range(args.num))
        missing = [i for i in expected if i not in d]
        n = len(d)
        succ = sum(1 for v in d.values() if v[0])
        score = sum(v[1] for v in d.values())
        ok = "" if not missing else f"missing={len(missing)}"
        rows.append((f"{task}__s{seed}", succ, n,
                     f"{succ / n * 100:.2f}" if n else "NA",
                     f"{score / n * 100:.2f}" if n else "NA", ok))
        totals[seed][0] += succ
        totals[seed][1] += n
        totals[seed][2] += score
        totals[seed][3] += 1

    out_path = run / "summary.tsv"
    with out_path.open("w") as f:
        f.write("task\tsuccess\tepisodes\tsuccess_rate\tscore\tnote\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
        grand = [0, 0, 0.0]
        for seed, (succ, n, score, _) in sorted(totals.items()):
            f.write(f"TOTAL__s{seed}\t{succ}\t{n}\t"
                    f"{succ / n * 100:.2f}\t{score / n * 100:.2f}\t\n")
            grand[0] += succ; grand[1] += n; grand[2] += score
        if grand[1]:
            f.write(f"TOTAL\t{grand[0]}\t{grand[1]}\t"
                    f"{grand[0] / grand[1] * 100:.2f}\t"
                    f"{grand[2] / grand[1] * 100:.2f}\t\n")
    print(f"[summarize] wrote {out_path}")
    print(out_path.read_text())


if __name__ == "__main__":
    main()
