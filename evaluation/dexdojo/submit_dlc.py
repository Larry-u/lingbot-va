#!/usr/bin/env python3
"""Submit a LingBot-VA DexDojo eval DLC job from the local dev machine.

Builds a PAI-DLC CreateJob body around evaluation/dexdojo/run_eval_lanes.sh
(the lane runner that mounts the shared CPFS and drives one
server+bridge+sim lane per GPU) and submits it via the local aliyun CLI
(OAuth profile; the shared `dlc` binary needs an AK we don't have).

Dry-run by default; pass --submit to create the job.

  python submit_dlc.py --ckpt /cpfs/yuxianggang/dexdojo_lingbot/checkpoints/checkpoint_step_1000 \
      --name lingbot-smoke --tasks insert_block --num 5 --seeds 0 --gpus 1
"""
import argparse
import json
import subprocess
import sys

WORKSPACE = "640957"
RESOURCE = "quotaedaioauy6hl"
IMAGE = ("dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai-training-algorithm/"
         "isaaclab:2.3.2-isaacsim5.1.0-py311-ubuntu24.04")
DATA_SOURCE = ("bmcpfs://cpfs-00000ub3ici1dnniit2i0-vpc-egtdgw.cn-hangzhou."
               "cpfs.aliyuncs.com/::/cpfs/")
EVAL_ROOT = "/cpfs/yuxianggang/dexdojo_lingbot"
REPO = f"{EVAL_ROOT}/code/lingbot-va"
DEXDOJO_ROOT = "/cpfs/share/dexdojo_eval/code/Xspark-wuji"
SIM_PY = "/cpfs/share/dexdojo_eval/venvs/luminis-fast/bin/python"
BRIDGE_PY = "/cpfs/share/dexdojo_eval/venvs/policy/bin/python"
LINGBOT_PY = f"{EVAL_ROOT}/venv/bin/python"
STATS = f"{EVAL_ROOT}/checkpoints/norm_stats.json"
BASE_MODEL = f"{EVAL_ROOT}/lingbot-va-base"
ALL_TASKS = ["collect_objects", "dual_bottles_pick", "hammer_beat",
             "insert_block", "retrieve_gap", "stack_bowls"]


def build_body(args):
    tasks = ",".join(ALL_TASKS) if args.tasks == "all" else args.tasks
    gpus = args.gpus.split(",")
    script = f"""set -euo pipefail
mkdir -p {EVAL_ROOT}/results
export DEXDOJO_ROOT={DEXDOJO_ROOT}
export DEXDOJO_SIM_PYTHON={SIM_PY}
export BRIDGE_PYTHON={BRIDGE_PY}
export LINGBOT_PYTHON={LINGBOT_PY}
export LINGBOT_DEXDOJO_STATS={STATS}
export BASE_MODEL_PATH={BASE_MODEL}
export HOME=/root
bash {REPO}/evaluation/dexdojo/run_eval_lanes.sh \\
  --ckpt {args.ckpt} \\
  --out {EVAL_ROOT}/results/{args.name} \\
  --tasks {tasks} --num {args.num} --seeds {args.seeds} --gpus {args.gpus} \\
  --extra-sim-args "{args.sim_args}"
"""
    return {
        "DisplayName": args.name,
        "JobType": "PyTorchJob",
        "WorkspaceId": int(WORKSPACE),
        "ResourceId": RESOURCE,
        "ExecutorInstances": 1,
        "ExecutorCores": int(args.cpu),
        "ExecutorMemory": args.memory,
        "ExecutorGpuCount": len(gpus),
        "ExecutorGpuType": "5090",
        "ImageUri": IMAGE,
        "DataSources": [{"DataSourceUri": DATA_SOURCE, "MountPath": "/cpfs",
                         "RoleArn": "", "MountPathProtocol": "bmcpfs"}],
        "Envs": [{"Name": "DEXDOJO_BACKEND", "Value": "dlc"},
                 {"Name": "OMNI_KIT_ACCEPT_EULA", "Value": "YES"}],
        "UserCommand": script,
        "MaxRunningTime": args.max_minutes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="CPFS checkpoint dir")
    ap.add_argument("--name", required=True, help="run name (results subdir)")
    ap.add_argument("--tasks", default="all",
                    help="'all' or comma list (default all)")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="logical GPU ids inside the worker")
    ap.add_argument("--cpu", type=int, default=96)
    ap.add_argument("--memory", default="800Gi")
    ap.add_argument("--sim-args", default="")
    ap.add_argument("--max-minutes", type=int, default=2160)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    body = build_body(args)
    print(json.dumps(body, indent=2, ensure_ascii=False))
    if not args.submit:
        print("\n[dry-run] pass --submit to create the job")
        return
    out = subprocess.run(
        ["aliyun", "--region", "cn-hangzhou", "pai-dlc", "CreateJob",
         "--body", json.dumps(body)],
        capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        sys.exit(out.returncode)


if __name__ == "__main__":
    main()
