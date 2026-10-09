#!/usr/bin/env python3
"""Submit a LingBot-VA DexDojo eval DLC job.

Routes through the shared dlc CLI on the wed_server DSW (its implicit
credential has PaiDLC CreateJob on workspace 640957; the OpenAPI CreateJob
schema fight isn't worth it). The job body wraps
evaluation/dexdojo/run_eval_lanes.sh: one worker, N GPUs = N lanes of
(server + bridge + sim).

Dry-run by default; --submit to create the job.

  python submit_dlc.py --ckpt .../checkpoint_step_1000 --name lingbot-e2e \
      --tasks insert_block --num 5 --seeds 0 --gpus 1 --submit
"""
import argparse
import shlex
import subprocess
import sys

SSH_HOST = "wed_server"
DLC_BIN = "/cpfs/share/dexdojo_eval/bin/dlc"
WORKSPACE = "640957"
QUOTA = "quotaedaioauy6hl"
IMAGE = ("dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai-training-algorithm/"
         "isaaclab:2.3.2-isaacsim5.1.0-py311-ubuntu24.04")
CPFS_URI = ("bmcpfs://cpfs-00000ub3ici1dnniit2i0-vpc-egtdgw.cn-hangzhou."
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


def build_remote_command(args, gpus):
    tasks = ",".join(ALL_TASKS) if args.tasks == "all" else args.tasks
    job_cmd = f"""set -euo pipefail
export DEXDOJO_ROOT={DEXDOJO_ROOT}
export DEXDOJO_SIM_PYTHON={SIM_PY}
export BRIDGE_PYTHON={BRIDGE_PY}
export LINGBOT_PYTHON={LINGBOT_PY}
export LINGBOT_DEXDOJO_STATS={STATS}
export BASE_MODEL_PATH={BASE_MODEL}
bash {REPO}/evaluation/dexdojo/run_eval_lanes.sh \\
  --ckpt {shlex.quote(args.ckpt)} \\
  --out {EVAL_ROOT}/results/{shlex.quote(args.name)} \\
  --tasks {tasks} --num {args.num} --seeds {args.seeds} --gpus {gpus} \\
  --extra-sim-args {shlex.quote(args.sim_args)}
"""
    import base64
    b64 = base64.b64encode(job_cmd.encode()).decode()
    return f"echo {b64} | base64 -d | bash"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="CPFS checkpoint dir")
    ap.add_argument("--name", required=True, help="run name (results subdir)")
    ap.add_argument("--tasks", default="all", help="'all' or comma list")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", type=int, default=8,
                    help="worker GPU count = lane count")
    ap.add_argument("--cpu-per-gpu", type=int, default=16)
    ap.add_argument("--mem-per-gpu", type=int, default=100,
                    help="GiB per GPU")
    ap.add_argument("--sim-args", default="")
    ap.add_argument("--max-minutes", type=int, default=2880)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    cmd = build_remote_command(args, args.gpus)
    dlc_args = [
        DLC_BIN, "submit", "pytorchjob",
        "-n", args.name,
        "-w", WORKSPACE,
        "--resource_id", QUOTA,
        "--worker_image", IMAGE,
        "--workers", "1",
        "--worker_gpu", str(args.gpus),
        "--worker_cpu", str(args.gpus * args.cpu_per_gpu),
        "--worker_memory", f"{args.gpus * args.mem_per_gpu}Gi",
        "--worker_shared_memory", "64Gi",
        "--job_max_running_time_minutes", str(args.max_minutes),
        "--data_source_uris", CPFS_URI,
        "--envs", "OMNI_KIT_ACCEPT_EULA=YES",
        "--command", cmd,
    ]
    remote = " ".join(shlex.quote(a) for a in dlc_args)
    print(f"[submit] ssh {SSH_HOST} -- {remote[:200]}...")
    if not args.submit:
        print("[dry-run] pass --submit to create the job")
        return
    out = subprocess.run(["ssh", SSH_HOST, remote],
                         capture_output=True, text=True)
    print(out.stdout[-2000:])
    if out.returncode != 0:
        print(out.stderr[-2000:], file=sys.stderr)
        sys.exit(out.returncode)


if __name__ == "__main__":
    main()
