#!/usr/bin/env bash
# set -euo pipefail

# CIRR training launcher using resized dataset paths and generated segmasks.
# Mirrors train.py CLI and saves best metrics/model automatically for CIRR.

# Root paths (ensure trailing slash for cirr_path)
PROJECT_ROOT="/DATA/home/ljq/Projects/OFFSET"
CIRR_ROOT="/DATA/home/ljq/Projects/OFFSET/data/CIRR/"  # must end with '/'

# Optional: pin to specific GPU (e.g., 0). Leave empty to use default.
GPU_ID=""
if [[ -n "$GPU_ID" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi

# Training configuration (adjust as needed)
LOCAL_RANK="-1"

OPTIMIZER="adamw"
BATCH_SIZE=2
NUM_EPOCHS=10
EPS=1e-8
WEIGHT_DECAY=1e-2
DROPOUT_RATE=0.5
HIDDEN_DIM=1024

P=4
Q=8
TAU=0.1
LAMBDA=1.0
ETA=1.0
MU=0.1
NU=10
KAPPA=0.5

SEED=42
LR=1e-4
CLIP_LR=1e-5

BACKBONE="ViT-H-14"

LR_DECAY=5
LR_DIV=0.1
MAX_DECAY_EPOCH=10
TOLERANCE_EPOCH=5
IF_SAVE=0

MODEL_DIR="${PROJECT_ROOT}/checkpoints"
SAVE_SUMMARY_STEPS=5
NUM_WORKERS=8
I_TAG="cirr-0"
NO_AMP="--no_amp"  # set to empty string to enable AMP

echo "[INFO] Starting CIRR training with root: ${CIRR_ROOT}"

python -u "${PROJECT_ROOT}/train.py" \
  --local_rank "${LOCAL_RANK}" \
  --dataset "cirr" \
  --fashioniq_path "" \
  --shoes_path "" \
  --cirr_path "${CIRR_ROOT}" \
  --optimizer "${OPTIMIZER}" \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --eps ${EPS} \
  --weight_decay ${WEIGHT_DECAY} \
  --dropout_rate ${DROPOUT_RATE} \
  --hidden_dim ${HIDDEN_DIM} \
  --P ${P} \
  --Q ${Q} \
  --tau_ ${TAU} \
  --lambda_ ${LAMBDA} \
  --eta_ ${ETA} \
  --mu_ ${MU} \
  --nu_ ${NU} \
  --kappa_ ${KAPPA} \
  --seed ${SEED} \
  --lr ${LR} \
  --clip_lr ${CLIP_LR} \
  --backbone "${BACKBONE}" \
  --lr_decay ${LR_DECAY} \
  --lr_div ${LR_DIV} \
  --max_decay_epoch ${MAX_DECAY_EPOCH} \
  --tolerance_epoch ${TOLERANCE_EPOCH} \
  --ifSave ${IF_SAVE} \
  --model_dir "${MODEL_DIR}" \
  --save_summary_steps ${SAVE_SUMMARY_STEPS} \
  --num_workers ${NUM_WORKERS} \
  --i "${I_TAG}" \
  ${NO_AMP}

echo "[DONE] CIRR training finished. Check outputs under ${MODEL_DIR}"