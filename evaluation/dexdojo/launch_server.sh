#!/usr/bin/env bash
# Launch one LingBot-VA policy server (DexDojo 54-D config) on a single GPU.
#
# usage: bash launch_server.sh <gpu_id> <port> [save_root]
#   gpu_id  CUDA device for the model
#   port    websocket port the bridge connects to
#
# The checkpoint's transformer/config.json must have attn_mode "torch" (or
# "flashattn") for inference — training saves "flex". fix_attn_mode below
# rewrites it in place if needed.
set -euo pipefail

GPU_ID=${1:?gpu id}
PORT=${2:?port}
SAVE_ROOT=${3:-/tmp/lingbot_va_server}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint dir}
export LINGBOT_DEXDOJO_MODEL=$MODEL_PATH
export LINGBOT_DEXDOJO_STATS=${LINGBOT_DEXDOJO_STATS:?set LINGBOT_DEXDOJO_STATS to norm_stats.json}

# training checkpoints carry attn_mode=flex which breaks inference
if grep -q '"attn_mode": *"flex"' "$MODEL_PATH/transformer/config.json"; then
  sed -i 's/"attn_mode": *"flex"/"attn_mode": "torch"/' \
    "$MODEL_PATH/transformer/config.json"
  echo "[launch] rewrote attn_mode flex -> torch in $MODEL_PATH"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=$GPU_ID TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=1 --master_port=$((29600 + GPU_ID % 100)) \
  -m wan_va.wan_va_server --config-name dexdojo \
  --port "$PORT" --save_root "$SAVE_ROOT"
