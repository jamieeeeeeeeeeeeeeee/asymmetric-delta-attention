#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLA_DIR="$REPO_ROOT/3rdparty/flash-linear-attention"

MODEL=""
SIZE=""
DATA_DIR=""
NGPU=8
OUTPUT=""
LOAD_STEP=-1
BATCH_SIZE="${BATCH_SIZE:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2" ;;
    --size) SIZE="$2" ;;
    --data-dir) DATA_DIR="$2" ;;
    --ngpu) NGPU="$2" ;;
    --output) OUTPUT="$2" ;;
    --wandb-project) export WANDB_PROJECT="$2" ;;
    --load-step) LOAD_STEP="$2" ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
  shift 2
done

if [ "$SIZE" = "340M" ]; then
  STEPS=30000
  DEFAULT_BATCH_SIZE=16
  CHECKPOINT_INTERVAL=3000
elif [ "$SIZE" = "1B" ]; then
  STEPS=95000
  DEFAULT_BATCH_SIZE=8
  CHECKPOINT_INTERVAL=5000
else
  echo "--size must be 340M or 1B"
  exit 1
fi

CONFIG="$REPO_ROOT/configs/${MODEL}_${SIZE}.json"
if [ ! -f "$CONFIG" ]; then
  echo "No config: $CONFIG"
  exit 1
fi

NUM_FILES=$(ls "$DATA_DIR"/train-*.parquet | wc -l)
if [ "$NUM_FILES" -ne 32 ]; then
  echo "Expected 32 parquet files in $DATA_DIR, found $NUM_FILES"
  exit 1
fi

if [ -z "$OUTPUT" ]; then
  OUTPUT="$REPO_ROOT/exp/${MODEL}_${SIZE}"
fi
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=$DEFAULT_BATCH_SIZE
fi

export NNODE="${NNODE:-1}"
export NGPU
export LOG_RANK=0
export WANDB_NAME="${WANDB_NAME:-${MODEL}_${SIZE}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${MODEL}_${SIZE}}"
TOTAL_GPUS=$((NNODE * NGPU))

# ensure 256 sequences per optimiser step whatever the GPU count
if [ $((256 % (BATCH_SIZE * TOTAL_GPUS))) -ne 0 ]; then
  echo "256 is not divisible by batch size $BATCH_SIZE x $TOTAL_GPUS GPUs"
  exit 1
fi
ACCUM=$((256 / (BATCH_SIZE * TOTAL_GPUS)))

# flame stops streaming if GPUs x workers > parquet files
if [ $((32 % TOTAL_GPUS)) -ne 0 ]; then
  echo "$TOTAL_GPUS GPUs must divide the 32 parquet files"
  exit 1
fi
NUM_WORKERS=$((32 / TOTAL_GPUS))
if [ "$NUM_WORKERS" -gt 16 ]; then
  NUM_WORKERS=16
fi

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$FLA_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$FLA_DIR"
fi
FLA_PATH=$(python -c "import fla; print(fla.__file__)")
if [[ "$FLA_PATH" != "$FLA_DIR"/* ]]; then
  echo "Wrong fla imported: $FLA_PATH"
  exit 1
fi

echo "model=$MODEL size=$SIZE gpus=$TOTAL_GPUS batch=$BATCH_SIZE accum=$ACCUM workers=$NUM_WORKERS steps=$STEPS output=$OUTPUT"

# actual training run!
cd "$REPO_ROOT/3rdparty/flame"
bash train.sh \
  --job.config_file "$REPO_ROOT/experiments/fla_noprof.toml" \
  --job.dump_folder "$OUTPUT" \
  --model.config "$CONFIG" \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.batch_size "$BATCH_SIZE" \
  --training.seq_len 2048 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps "$ACCUM" \
  --training.steps "$STEPS" \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree "$TOTAL_GPUS" \
  --training.data_parallel_shard_degree 1 \
  --training.tensor_parallel_degree 1 \
  --training.dataset parquet \
  --training.data_dir "$DATA_DIR" \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers "$NUM_WORKERS" \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --optimizer.name AdamW \
  --optimizer.lr 4e-4 \
  --optimizer.eps 1e-8 \
  --optimizer.weight_decay 0.01 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --checkpoint.interval "$CHECKPOINT_INTERVAL" \
  --checkpoint.load_step "$LOAD_STEP" \
  --metrics.log_freq 10
