#!/bin/bash

# Pretraining sweep over model scale x Laplacian head count x random seed.
#
# Usage:
#   bash runs/laplacian_sweep.sh [label]
#
# Everything is env-overridable, e.g.:
#   DEPTHS="12 20" LAPLACIAN_HEADS="0 2" SEEDS="1 2 3" bash runs/laplacian_sweep.sh mysweep
#
# By default each cell trains to its own compute-optimal horizon (--target-param-data-ratio).
# Set FLOPS_BUDGETS to sweep at fixed compute budgets instead (like runs/scaling_laws.sh):
#   FLOPS_BUDGETS="1e18 4.64e18" bash runs/laplacian_sweep.sh
#
# The sweep is resume-safe: cells already present in results.csv are skipped, so you can
# re-run this after a preemption and it picks up where it left off.

set -u

LABEL="${1:-${LABEL:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"

# Sweep axes
DEPTHS="${DEPTHS:-12 16 20}"
LAPLACIAN_HEADS="${LAPLACIAN_HEADS:-0 1 2 4}"
SEEDS="${SEEDS:-1 2 3}"
# Empty => compute-optimal horizon per depth. Non-empty => sweep these fixed FLOPs budgets.
FLOPS_BUDGETS="${FLOPS_BUDGETS:-}"
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-12}"

# Hardware / logging
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-laplacian_${LABEL}}"
EVAL_TOKENS="${EVAL_TOKENS:-$((100 * 524288))}"  # ~100M tokens for the final val eval
# Extra args appended verbatim to every base_train invocation (e.g. "--no-ve --window-pattern=L")
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
[ -f .venv/bin/activate ] && source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/laplacian_sweep_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,laplacian_heads,seed,model_tag,model_dim,params_wte,params_value_embeds,params_lm_head,params_transformer,params_scalars,params_total,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# A cell is identified by (flops_budget, depth, laplacian_heads, seed) -- the first four columns
run_exists() {
    grep -q "^${1},${2},${3},${4}," "$RESULTS_FILE" 2>/dev/null
}

# Reduce --device-batch-size to avoid OOM at larger depths
device_batch_size_arg() {
    local d=$1
    if [ "$d" -ge 28 ]; then echo "--device-batch-size=8"
    elif [ "$d" -ge 20 ]; then echo "--device-batch-size=16"
    else echo "--device-batch-size=32"
    fi
}

# Pull a value out of the padded "Parameter counts:" table, e.g. "wte : 25,165,824"
# Matches "^key " so that e.g. "total" does not also match "params_total"
scrape_param() {
    grep "^$2 " "$1" | tail -1 | grep -oP '[\d,]+' | tr -d ','
}

# Iterate FLOPs budgets, or a single sentinel meaning "compute-optimal"
if [ -z "$FLOPS_BUDGETS" ]; then
    BUDGET_LIST="optimal"
else
    BUDGET_LIST="$FLOPS_BUDGETS"
fi

for flops in $BUDGET_LIST; do
for d in $DEPTHS; do
for lap in $LAPLACIAN_HEADS; do
for seed in $SEEDS; do

    if run_exists "$flops" "$d" "$lap" "$seed"; then
        log "Skipping d=$d lap=$lap seed=$seed at $flops (already in results)"
        continue
    fi

    log "=============================================="
    log "Training d=$d laplacian_heads=$lap seed=$seed budget=$flops"
    log "=============================================="

    if [ "$flops" = "optimal" ]; then
        HORIZON_ARGS="--target-param-data-ratio=$PARAM_DATA_RATIO"
    else
        HORIZON_ARGS="--target-flops=$flops --target-param-data-ratio=-1"
    fi

    # base_train derives the model tag from the architecture + seed, so cells never collide.
    # We use our own slug only to name the log file.
    SLUG="d${d}_lap${lap//,/x}_s${seed}"
    LOG_FILE="$RESULTS_DIR/${SLUG}_train.log"

    START_TIME=$(date +%s)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$d \
        --laplacian-heads=$lap \
        --seed=$seed \
        $HORIZON_ARGS \
        --run="${WANDB_RUN}_${SLUG}" \
        --eval-tokens=$EVAL_TOKENS \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        $(device_batch_size_arg "$d") \
        $EXTRA_TRAIN_ARGS \
        2>&1 | tee "$LOG_FILE"
    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # The tag base_train actually resolved to (used to locate checkpoints / eval outputs)
    MODEL_TAG=$(grep "Resolved model tag:" "$LOG_FILE" | tail -1 | sed 's/.*: //')

    PARAMS_WTE=$(scrape_param "$LOG_FILE" wte)
    PARAMS_VE=$(scrape_param "$LOG_FILE" value_embeds)
    PARAMS_LM=$(scrape_param "$LOG_FILE" lm_head)
    PARAMS_TRANSFORMER=$(scrape_param "$LOG_FILE" transformer_matrices)
    PARAMS_SCALARS=$(scrape_param "$LOG_FILE" scalars)
    PARAMS_TOTAL=$(scrape_param "$LOG_FILE" total)

    NUM_ITERS=$(grep "number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    BATCH_SIZE=$(grep "Total batch size" "$LOG_FILE" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
    if [ -n "$NUM_ITERS" ] && [ -n "$BATCH_SIZE" ]; then
        TOKENS_TRAINED=$((NUM_ITERS * BATCH_SIZE))
    else
        log "WARNING: could not scrape iterations/batch size for $SLUG"
        TOKENS_TRAINED=0
    fi
    MODEL_DIM=$(grep -oP '"n_embd":\s*\K\d+' "$LOG_FILE" | head -1)
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    if [ -z "$CORE_SCORE" ]; then
        log "WARNING: could not extract CORE score for $SLUG"
        CORE_SCORE="0.0"
    fi

    log "  tag=$MODEL_TAG params=$PARAMS_TOTAL iters=$NUM_ITERS bpb=$VAL_BPB CORE=$CORE_SCORE time=${TRAIN_TIME}s"

    echo "$flops,$d,$lap,$seed,$MODEL_TAG,$MODEL_DIM,$PARAMS_WTE,$PARAMS_VE,$PARAMS_LM,$PARAMS_TRANSFORMER,$PARAMS_SCALARS,$PARAMS_TOTAL,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"

done
done
done
done

log "=============================================="
log "Laplacian Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
