#!/usr/bin/env bash
# Lean DexDojo closed-loop eval lane runner for LingBot-VA (54-D DexDojo SFT).
#
# Designed to run INSIDE one DLC worker (or any GPU host that mounts the
# shared /cpfs): one lane per GPU = LingBot-VA server (GPU) + lingbot bridge
# (CPU, XPolicyLab protocol) + Isaac sim (GPU, num_envs=1). Work items are
# (task, seed) pairs consumed from a shared queue via flock; the sim's own
# resume manifests make reruns continue in place.
#
# Required environment:
#   DEXDOJO_ROOT         Xspark-wuji checkout (shared, read-only usage)
#   DEXDOJO_SIM_PYTHON   Isaac sim env python (shared luminis-fast venv)
#   BRIDGE_PYTHON        XPolicyLab policy env python (shared policy venv)
#   LINGBOT_PYTHON       this repo's serving venv python
#   LINGBOT_DEXDOJO_STATS  norm_stats.json path
#   BASE_MODEL_PATH      lingbot-va base dir (vae/tokenizer/text_encoder)
#
# usage: bash run_eval_lanes.sh --ckpt <dir> --out <run_dir> \
#          [--tasks all|t1,t2] [--num 50] [--seeds 0,1,2] [--gpus 0,1,...] \
#          [--base-port 21000] [--extra-sim-args "..."]
set -uo pipefail

CKPT="" OUT="" TASKS="all" NUM=50 SEEDS="0,1,2" GPUS="" BASE_PORT=21000 EXTRA_SIM_ARGS="" SERVER_GPUS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --ckpt) CKPT=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --tasks) TASKS=$2; shift 2;;
    --num) NUM=$2; shift 2;;
    --seeds) SEEDS=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --server-gpus) SERVER_GPUS=$2; shift 2;;
    --base-port) BASE_PORT=$2; shift 2;;
    --extra-sim-args) EXTRA_SIM_ARGS=$2; shift 2;;
    *) echo "[lanes] unknown arg $1"; exit 1;;
  esac
done
: "${CKPT:?--ckpt required}" "${OUT:?--out required}"
: "${DEXDOJO_ROOT:?}" "${DEXDOJO_SIM_PYTHON:?}" "${BRIDGE_PYTHON:?}" "${LINGBOT_PYTHON:?}"
: "${LINGBOT_DEXDOJO_STATS:?}" "${BASE_MODEL_PATH:?}"
GPUS=${GPUS:-0}
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_ROOT=${EVAL_ROOT:-$(dirname "$(dirname "$REPO")")}

# The shared sim venv's editable installs point at /mnt/data/share/... (host
# layout); DLC pods mount the CPFS at /cpfs — bridge the two (same fix the
# w0 service applies inside its DLC command).
mkdir -p /mnt/data
for item in share xiaoxiong; do
  link=/mnt/data/$item
  target=/cpfs/$item
  if [ -e "$link" ] || [ -L "$link" ]; then
    [ "$(readlink -f "$link")" = "$target" ] || { echo "[lanes] $link exists but is not $target" >&2; exit 97; }
  else
    ln -s "$target" "$link"
  fi
done

ALL_TASKS="collect_objects,dual_bottles_pick,hammer_beat,insert_block,retrieve_gap,stack_bowls"
[[ "$TASKS" == "all" ]] && TASKS=$ALL_TASKS

mkdir -p "$OUT"/{logs,queue}
VIEW="$OUT/view"
mkdir -p "$VIEW"

# ---- view dir: everything symlinked to the shared checkout except env_cfg
# (num_envs forced to 1: one AR stream per lane) and eval_result ------------
for entry in "$DEXDOJO_ROOT"/*; do
  name=$(basename "$entry")
  [[ "$name" == "env_cfg" || "$name" == "eval_result" ]] && continue
  [[ -e "$VIEW/$name" ]] || ln -s "$entry" "$VIEW/$name"
done
mkdir -p "$VIEW/env_cfg/sim" "$VIEW/eval_result"
for f in "$DEXDOJO_ROOT"/env_cfg/*.yml; do
  [[ -e "$VIEW/env_cfg/$(basename "$f")" ]] || cp "$f" "$VIEW/env_cfg/"
done
for d in camera robot scene; do
  [[ -e "$VIEW/env_cfg/$d" ]] || cp -r "$DEXDOJO_ROOT/env_cfg/$d" "$VIEW/env_cfg/$d"
done
cp "$DEXDOJO_ROOT"/env_cfg/sim/*.yml "$VIEW/env_cfg/sim/"
sed -i 's/num_envs: *[0-9]*/num_envs: 1/' "$VIEW"/env_cfg/sim/*.yml
echo "[lanes] env_cfg num_envs forced to 1 in $VIEW/env_cfg/sim"

# ---- work queue --------------------------------------------------------------
QUEUE="$OUT/queue/items.tsv"
LOCK="$OUT/queue/lock"
: > "$QUEUE"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
IFS=',' read -ra TASK_LIST <<< "$TASKS"
for t in "${TASK_LIST[@]}"; do
  for s in "${SEED_LIST[@]}"; do
    echo -e "${t}\t${s}" >> "$QUEUE"
  done
done

# node-local Kit portable root (same scoping as the w0 service)
KIT_PORTABLE_ROOT="${LUMINIS_KIT_PORTABLE_ROOT:-$HOME/.cache/lingbot-kit-$(echo "$DEXDOJO_SIM_PYTHON" | md5sum | cut -c1-8)}"
mkdir -p "$KIT_PORTABLE_ROOT" "$HOME/.cache/stub-bin"
printf '#!/bin/sh\nexit 0\n' > "$HOME/.cache/stub-bin/zenity"
chmod +x "$HOME/.cache/stub-bin/zenity"
export LUMINIS_KIT_PORTABLE_ROOT

# ---- per-lane processes -------------------------------------------------------
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_server() {  # $1 gpu  $2 port
  nohup bash -c "cd '$REPO' && CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LINGBOT_DEXDOJO_MODEL='$CKPT' LINGBOT_DEXDOJO_STATS='$LINGBOT_DEXDOJO_STATS' \
  LINGBOT_DEXDOJO_BASE='$BASE_MODEL_PATH' \
  exec '$LINGBOT_PYTHON' -m wan_va.wan_va_server \
      --config-name dexdojo --port '$2' \
      --save_root '$OUT/logs/server_$1'" \
      > "$OUT/logs/server_$1.log" 2>&1 &
  PIDS+=($!)
}

start_bridge() {  # $1 gpu  $2 bridge_port  $3 server_port
  DEXDOJO_ROOT="$DEXDOJO_ROOT" \
  nohup "$BRIDGE_PYTHON" "$REPO/evaluation/dexdojo/lingbot_bridge.py" \
      --port "$2" --server-url "ws://127.0.0.1:$3" \
      --dexdojo-root "$DEXDOJO_ROOT" \
      > "$OUT/logs/bridge_$1.log" 2>&1 &
  PIDS+=($!)
}

wait_port() {  # $1 port  $2 label
  for _ in $(seq 1 720); do
    if curl -s -o /dev/null "http://127.0.0.1:$1/healthz" 2>/dev/null; then
      echo "[lanes] $2 up on :$1"; return 0
    fi
    sleep 5
  done
  echo "[lanes] TIMEOUT waiting for $2 on :$1"; return 1
}

wait_bridge() {  # $1 port  $2 label  (no HTTP healthz; poll TCP)
  for _ in $(seq 1 240); do
    if "$BRIDGE_PYTHON" -c "import socket;s=socket.socket();s.settimeout(1);exit(0 if s.connect_ex(('127.0.0.1',$1))==0 else 1)" 2>/dev/null; then
      echo "[lanes] $2 up on :$1"; return 0
    fi
    sleep 5
  done
  echo "[lanes] TIMEOUT waiting for $2 on :$1"; return 1
}

run_lane() {  # $1 gpu  $2 lane_idx  $3 server_gpu
  local gpu=$1 lane_idx=$2 server_gpu=${3:-$1}
  local sport=$((BASE_PORT + lane_idx)) bport=$((BASE_PORT + 1000 + lane_idx))
  # ckpt must be servable BEFORE the server starts: link the base's
  # vae/tokenizer/text_encoder in and flip attn_mode to torch
  for sub in vae tokenizer text_encoder; do
    [[ -e "$CKPT/$sub" ]] || ln -s "$BASE_MODEL_PATH/$sub" "$CKPT/$sub"
  done
  sed -i 's/"attn_mode": *"flex"/"attn_mode": "torch"/' "$CKPT/transformer/config.json" 2>/dev/null || true
  start_server "$server_gpu" "$sport"
  if ! wait_port "$sport" "server(gpu$gpu)"; then return 1; fi
  start_bridge "$gpu" "$bport" "$sport"
  if ! wait_bridge "$bport" "bridge(gpu$gpu)"; then return 1; fi

  local task seed item_rc done_marker
  while :; do
    item=$(flock "$LOCK" bash -c "head -1 '$QUEUE' 2>/dev/null; sed -i '1d' '$QUEUE' 2>/dev/null")
    [[ -z "$item" ]] && break
    task=$(echo "$item" | cut -f1); seed=$(echo "$item" | cut -f2)
    echo "[lane $gpu] START $task s$seed $(date +%H:%M:%S)"
    (
      cd "$VIEW"
      # eval_policy.sh runs bare `python`; the Isaac sim venv must own it.
      # Also prepend the zenity stub so Isaac's crash handler cannot wedge,
      # and the pyarrow shadow: the shared sim venv's pyarrow 25 is missing
      # pyarrow.vendored (breaks pandas -> open3d imports); an older wheel
      # on PYTHONPATH shadows it cleanly.
      export PATH="$HOME/.cache/stub-bin:$(dirname "$DEXDOJO_SIM_PYTHON"):${PATH}"
      export PYTHONPATH="$EVAL_ROOT/sim_pyarrow_fix:${PYTHONPATH:-}"
      export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
      setsid bash "$DEXDOJO_ROOT/scripts/eval_policy.sh" \
        --root_dir "$VIEW" \
        --task_name "$task" \
        --env_cfg_type "${ENV_CFG_TYPE:-tianji_marvin_wuji}" \
        --device_id "$gpu" \
        --policy_name vitra_w0 \
        --port "$bport" \
        --eval_batch false \
        --seed "$seed" \
        --additional_info "$(basename "$OUT")__${task}__s${seed}" \
        $EXTRA_SIM_ARGS \
        > "$OUT/logs/sim_${task}__s${seed}.log" 2>&1
    )
    item_rc=$?
    echo "[lane $gpu] END   $task s$seed rc=$item_rc $(date +%H:%M:%S)"
    if [[ $item_rc -ne 0 ]]; then
      echo -e "${task}\t${seed}" >> "$OUT/queue/failed.tsv"
    fi
  done
}

IFS=',' read -ra GPU_LIST <<< "$GPUS"
if [[ -n "$SERVER_GPUS" ]]; then
  IFS=',' read -ra SERVER_LIST <<< "$SERVER_GPUS"
else
  SERVER_LIST=()
fi
for i in "${!GPU_LIST[@]}"; do
  run_lane "${GPU_LIST[$i]}" "$i" "${SERVER_LIST[$i]:-}" &
  sleep 15   # stagger lane starts (init contention)
done
wait

"$LINGBOT_PYTHON" "$REPO/evaluation/dexdojo/summarize.py" --run "$OUT" --num "$NUM"
echo "[lanes] all done: $OUT"
