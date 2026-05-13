#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/launch_ddp.sh <config> <gpu_ids_csv>" >&2
  echo "Example: scripts/launch_ddp.sh configs/ptbxl_hifuse.yaml 0,1" >&2
  exit 2
fi

CONFIG="$1"
GPU_IDS="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$(python - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["output_dir"])
PY
)"
NPROC="$(python - "$GPU_IDS" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(",") if x.strip()]))
PY
)"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

LOG="$OUTPUT_DIR/train_ddp.log"
PIDFILE="$OUTPUT_DIR/train_ddp.pid"
STATUS="$OUTPUT_DIR/train_ddp.status"
SESSION_NAME="$(basename "$OUTPUT_DIR")_ddp"

screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
screen -dmS "$SESSION_NAME" bash -lc "
  set -euo pipefail
  echo running > '$STATUS'
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate pytorch
  export CUDA_VISIBLE_DEVICES=$GPU_IDS
  torchrun --standalone --nproc_per_node=$NPROC scripts/train.py --config '$CONFIG' > '$LOG' 2>&1
  code=\$?
  echo exit:\$code > '$STATUS'
  exit \$code
"

PID="$(screen -ls | awk -v s="$SESSION_NAME" '$0 ~ s {split($1,a,"."); print a[1]; exit}')"
echo "$PID" > "$PIDFILE"
echo "$PID"
