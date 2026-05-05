#!/bin/bash

# Launches LoRA fine-tuning on top of a frozen d30 base checkpoint, then runs LoRA eval.
#
# Targets the pretrained d30 models in
#   ${NANOCHAT_BASE_DIR}/base_checkpoints/d30_lap<N>_vanilla_ratio12/
# (n_layer=30, n_embd=1920, n_head=15, seq_len=2048, SSSL window).
#
# Usage:
#   bash runs/lora_sft.sh                              # default: lap0 base
#   bash runs/lora_sft.sh 12                           # use d30_lap12_vanilla_ratio12
#   bash runs/lora_sft.sh 0,2,2,0,...                  # custom per-layer Laplacian
#   LORA_RANK=32 LORA_LR=1e-4 bash runs/lora_sft.sh    # override hyperparameters
#   WANDB_RUN=d30_lora bash runs/lora_sft.sh           # enable wandb

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/home/yucz/scratch/nanochat_artifacts"
mkdir -p $NANOCHAT_BASE_DIR
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Optional first arg: Laplacian head spec, used to select which d30 base checkpoint to
# load. Available options under base_checkpoints/: lap{0,3,6,9,12,15}.
LAPLACIAN_HEADS="${1:-0}"
LAPLACIAN_HEADS_TAG="${LAPLACIAN_HEADS//,/x}"
PARAM_DATA_RATIO=12
DEPTH=30
# Note: d30 tags are "...vanilla_ratio12" (no "true_" prefix that the d24 retrains use).
RUN_TAG="d${DEPTH}_lap${LAPLACIAN_HEADS_TAG}_vanilla_ratio${PARAM_DATA_RATIO}"

# LoRA hyperparameters tuned for the d30 base (~1.5B params, n_embd=1920, 30 layers).
# Defaults below produce ~16.6M trainable adapter params (~1.1% of total) when
# applied to all 6 matmul matrices per block.
#
#   LORA_RANK=16, LORA_ALPHA=32 (alpha = 2*rank, scale = 2.0)
#     - Modern community default for SFT on 1-3B models (HF PEFT, Axolotl, Llama-Factory).
#     - LoRA paper used r=4-8 for GPT-3 (175B); for smaller models, r=16 has
#       negligible param-count cost while clearly outperforming r=8 on instruction SFT.
#   LORA_LR=2e-4
#     - AdamW + LoRA SFT sweet spot is 1e-4 to 5e-4 (well above full-FT's ~1e-5,
#       since adapters start near zero and need a higher learning rate to train).
#   LORA_DROPOUT=0.05
#     - Light regularization typical of modern SFT recipes.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_LR="${LORA_LR:-2e-4}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TAG="${LORA_TAG:-${RUN_TAG}_lora_r${LORA_RANK}_a${LORA_ALPHA}_lr${LORA_LR}}"

# Per-device batch size. d30 has ~25% more layers and ~2.5x the embedding dim of d24,
# so activation memory dominates even though the base weights are frozen. 4 is safe
# on H100 80GB; bump to 8 if you have headroom.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"

export NANOCHAT_MODEL_TAG="$LORA_TAG"

# -----------------------------------------------------------------------------
# Python venv setup with uv
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# [ -d ".venv" ] || uv venv
# uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (set WANDB_RUN=<name> to enable; default "dummy" disables wandb)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# SFT data: identity conversations file
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
#     https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# -----------------------------------------------------------------------------
# Multi-GPU launch (default). Comment out and use the single-GPU block below
# if you only have one GPU available.
# torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.lora_sft -- \
#     --device-batch-size="$DEVICE_BATCH_SIZE" \
#     --model-tag="$RUN_TAG" \
#     --output-tag="$LORA_TAG" \
#     --learning-rate="$LORA_LR" \
#     --weight-decay=0.0 \
#     --lora-rank="$LORA_RANK" \
#     --lora-alpha="$LORA_ALPHA" \
#     --lora-dropout="$LORA_DROPOUT" \
#     --eval-every -1 \
#     --run="$WANDB_RUN"

# torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.lora_eval -- \
#     -g "$LORA_TAG"

# -----------------------------------------------------------------------------
# Single-GPU launch (uncomment, and comment out the multi-GPU block above):
python -m scripts.lora_sft \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$RUN_TAG" \
    --output-tag="$LORA_TAG" \
    --learning-rate="$LORA_LR" \
    --weight-decay=0.0 \
    --lora-rank="$LORA_RANK" \
    --lora-alpha="$LORA_ALPHA" \
    --lora-dropout="$LORA_DROPOUT" \
    --eval-every -1 \
    --run="$WANDB_RUN"

python -m scripts.lora_eval -g "$LORA_TAG"

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
python -m nanochat.report generate --model-tag "$LORA_TAG"
