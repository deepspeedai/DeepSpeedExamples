#!/usr/bin/env bash
# End-to-end AutoEP vs ZeRO-3 leaf comparison workflow.
#
# Usage:
#   bash run_compare.sh [--num_gpus 8] [--autoep_size 8] [--steps 50] [--out_dir runs/autoep_results]
#
# This script:
#   1. Runs find_max_layers.py to determine largest feasible shared layer count
#   2. Generates shared init weights artifact
#   3. Runs train.py in both modes at the determined layer count
#   4. Generates comparison plots and summary via compare_metrics.py

set -euo pipefail

# Defaults
NUM_GPUS=8
AUTOEP_SIZE=8
MIN_LAYERS=2
MAX_LAYERS=64
STEPS=50
WARMUP_STEPS=5
TRIAL_STEPS=20
SEQ_LEN=128
MICRO_BATCH_SIZE=2
GRAD_ACCUM=1
SEED=42
MASTER_PORT=29600
OUT_DIR="runs/autoep_results/$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false
TRIAL_TIMEOUT=300
INIT_WEIGHTS_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --autoep_size) AUTOEP_SIZE="$2"; shift 2 ;;
        --min_layers) MIN_LAYERS="$2"; shift 2 ;;
        --max_layers) MAX_LAYERS="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --warmup_steps) WARMUP_STEPS="$2"; shift 2 ;;
        --trial_steps) TRIAL_STEPS="$2"; shift 2 ;;
        --seq_len) SEQ_LEN="$2"; shift 2 ;;
        --micro_batch_size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum) GRAD_ACCUM="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        --trial_timeout) TRIAL_TIMEOUT="$2"; shift 2 ;;
        --init_weights_path) INIT_WEIGHTS_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate args
if [[ $MIN_LAYERS -gt $MAX_LAYERS ]]; then
    echo "ERROR: min_layers ($MIN_LAYERS) > max_layers ($MAX_LAYERS)"
    exit 1
fi
if [[ $STEPS -le $WARMUP_STEPS ]]; then
    echo "ERROR: steps ($STEPS) must be > warmup_steps ($WARMUP_STEPS)"
    exit 1
fi
if [[ $NUM_GPUS -lt 2 ]]; then
    echo "ERROR: num_gpus ($NUM_GPUS) must be >= 2"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUT_DIR"

if [[ -z "$INIT_WEIGHTS_PATH" ]]; then
    INIT_WEIGHTS_PATH="$OUT_DIR/init_weights.safetensors"
fi
INIT_WEIGHTS="$INIT_WEIGHTS_PATH"

MANIFEST="$OUT_DIR/manifest.json"
LAYER_SEARCH_JSON="$OUT_DIR/layer_search.json"
AUTOEP_CSV="$OUT_DIR/metrics_autoep.csv"
ZERO3_CSV="$OUT_DIR/metrics_zero3_leaf.csv"
SUMMARY_JSON="$OUT_DIR/summary.json"

# Helper: atomic manifest write via Python
update_manifest() {
    python -c "
import json, os, sys
path = sys.argv[1]
if os.path.exists(path):
    with open(path) as f:
        m = json.load(f)
else:
    m = {'schema_version': 1, 'status': 'running', 'stages': [], 'artifacts': {},
         'started_at': None, 'finished_at': None, 'dry_run': False,
         'args': {}, 'final_layers': None, 'error': None}
exec(sys.argv[2])
tmp = path + '.tmp'
with open(tmp, 'w') as f:
    json.dump(m, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
dir_fd = os.open(os.path.dirname(path) or '.', os.O_DIRECTORY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
" "$1" "$2"
}

# Initialize manifest
update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['started_at'] = datetime.now(timezone.utc).isoformat()
m['args'] = {
    'num_gpus': $NUM_GPUS, 'autoep_size': $AUTOEP_SIZE, 'min_layers': $MIN_LAYERS, 'max_layers': $MAX_LAYERS,
    'steps': $STEPS, 'warmup_steps': $WARMUP_STEPS, 'trial_steps': $TRIAL_STEPS,
    'seq_len': $SEQ_LEN, 'micro_batch_size': $MICRO_BATCH_SIZE,
    'grad_accum': $GRAD_ACCUM, 'seed': $SEED, 'init_weights_path': '$INIT_WEIGHTS',
}
m['dry_run'] = $( [[ "$DRY_RUN" == "true" ]] && echo "True" || echo "False" )
"

# Cleanup trap: finalize manifest on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['status'] = 'failed'
m['finished_at'] = datetime.now(timezone.utc).isoformat()
m['error'] = 'Script exited with code $exit_code'
" || true
    fi
}
trap cleanup EXIT

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN ==="
    echo "Would run layer search, then train at L_final in both modes."
    update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['status'] = 'dry_run'
m['finished_at'] = datetime.now(timezone.utc).isoformat()
"
    echo "Manifest: $MANIFEST"
    exit 0
fi

echo "=== Step 1: Find max feasible layers ==="
update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'].append({
    'name': 'find_max_layers', 'status': 'running',
    'started_at': datetime.now(timezone.utc).isoformat(),
    'finished_at': None, 'exit_code': None, 'command': [], 'artifacts': {}
})
"

python find_max_layers.py \
    --min_layers "$MIN_LAYERS" \
    --max_layers "$MAX_LAYERS" \
    --seq_len "$SEQ_LEN" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --trial_steps "$TRIAL_STEPS" \
    --num_gpus "$NUM_GPUS" \
    --autoep_size "$AUTOEP_SIZE" \
    --seed "$SEED" \
    --master_port "$MASTER_PORT" \
    --trial_timeout "$TRIAL_TIMEOUT" \
    --output_json "$LAYER_SEARCH_JSON" \
    --log_dir "$OUT_DIR/layer_search_logs/"

# Extract L_final
L_FINAL=$(python -c "
import json, sys
with open('$LAYER_SEARCH_JSON') as f:
    d = json.load(f)
print(d['final_layers'])
")

if [[ "$L_FINAL" == "0" || -z "$L_FINAL" ]]; then
    echo "ERROR: No feasible layer count found."
    exit 1
fi

echo "=== L_final = $L_FINAL ==="

update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['final_layers'] = $L_FINAL
m['stages'][-1]['status'] = 'success'
m['stages'][-1]['finished_at'] = datetime.now(timezone.utc).isoformat()
m['stages'][-1]['exit_code'] = 0
m['artifacts']['layer_search_json'] = '$LAYER_SEARCH_JSON'
"

echo "=== Step 2: Prepare shared init weights artifact ==="
update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'].append({
    'name': 'prepare_init_weights', 'status': 'running',
    'started_at': datetime.now(timezone.utc).isoformat(),
    'finished_at': None, 'exit_code': None, 'command': [], 'artifacts': {}
})
"

python prepare_init_weights.py \
    --model qwen3_5 \
    --num_layers "$L_FINAL" \
    --seed "$SEED" \
    --output "$INIT_WEIGHTS"

INIT_WEIGHTS_SHA256=$(python -c "
import hashlib
path = '$INIT_WEIGHTS'
h = hashlib.sha256()
with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
")

update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'][-1]['status'] = 'success'
m['stages'][-1]['finished_at'] = datetime.now(timezone.utc).isoformat()
m['stages'][-1]['exit_code'] = 0
m['stages'][-1]['artifacts']['init_weights_path'] = '$INIT_WEIGHTS'
m['stages'][-1]['artifacts']['init_weights_sha256'] = '$INIT_WEIGHTS_SHA256'
m['artifacts']['init_weights_path'] = '$INIT_WEIGHTS'
m['artifacts']['init_weights_sha256'] = '$INIT_WEIGHTS_SHA256'
"

echo "=== Step 3a: Final comparison - AutoEP ==="
update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'].append({
    'name': 'train_autoep', 'status': 'running',
    'started_at': datetime.now(timezone.utc).isoformat(),
    'finished_at': None, 'exit_code': None, 'command': [], 'artifacts': {}
})
"

AUTOEP_PORT=$((MASTER_PORT + 100))
deepspeed --num_gpus "$NUM_GPUS" --master_port "$AUTOEP_PORT" \
    train.py \
    --mode autoep \
    --autoep_size "$AUTOEP_SIZE" \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --num_layers "$L_FINAL" \
    --seq_len "$SEQ_LEN" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --seed "$SEED" \
    --load_init_weights "$INIT_WEIGHTS" \
    --metrics_out "$AUTOEP_CSV"

update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'][-1]['status'] = 'success'
m['stages'][-1]['finished_at'] = datetime.now(timezone.utc).isoformat()
m['stages'][-1]['exit_code'] = 0
m['artifacts']['autoep_csv'] = '$AUTOEP_CSV'
"

echo "=== Step 3b: Final comparison - ZeRO-3 leaf ==="
update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'].append({
    'name': 'train_zero3_leaf', 'status': 'running',
    'started_at': datetime.now(timezone.utc).isoformat(),
    'finished_at': None, 'exit_code': None, 'command': [], 'artifacts': {}
})
"

ZERO3_PORT=$((MASTER_PORT + 200))
deepspeed --num_gpus "$NUM_GPUS" --master_port "$ZERO3_PORT" \
    train.py \
    --mode zero3_leaf \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --num_layers "$L_FINAL" \
    --seq_len "$SEQ_LEN" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --seed "$SEED" \
    --load_init_weights "$INIT_WEIGHTS" \
    --metrics_out "$ZERO3_CSV"

update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'][-1]['status'] = 'success'
m['stages'][-1]['finished_at'] = datetime.now(timezone.utc).isoformat()
m['stages'][-1]['exit_code'] = 0
m['artifacts']['zero3_leaf_csv'] = '$ZERO3_CSV'
"

echo "=== Step 4: Generate comparison ==="
update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'].append({
    'name': 'compare_metrics', 'status': 'running',
    'started_at': datetime.now(timezone.utc).isoformat(),
    'finished_at': None, 'exit_code': None, 'command': [], 'artifacts': {}
})
"

python compare_metrics.py \
    --autoep_csv "$AUTOEP_CSV" \
    --zero3_leaf_csv "$ZERO3_CSV" \
    --out_dir "$OUT_DIR" \
    --out_json "$SUMMARY_JSON" \
    --autoep_label "AutoEP + ZeRO-1" \
    --zero3_leaf_label "HF + ZeRO-3 leaf" \
    --warmup_steps "$WARMUP_STEPS"

update_manifest "$MANIFEST" "
from datetime import datetime, timezone
m['stages'][-1]['status'] = 'success'
m['stages'][-1]['finished_at'] = datetime.now(timezone.utc).isoformat()
m['stages'][-1]['exit_code'] = 0
m['artifacts']['summary_json'] = '$SUMMARY_JSON'
m['status'] = 'success'
m['finished_at'] = datetime.now(timezone.utc).isoformat()
"

# Clear the exit trap since we succeeded
trap - EXIT

echo ""
echo "=== Complete ==="
echo "Results: $OUT_DIR"
echo "Summary: $SUMMARY_JSON"
echo "Manifest: $MANIFEST"
