#!/usr/bin/env bash
# set -euo pipefail

# FashionIQ training launcher using resized dataset paths.
# Runs training for each category (dress, shirt, toptee) with all parameters explicitly set.

# Root paths (ensure trailing slash for fashioniq_path)
PROJECT_ROOT="/DATA/home/ljq/Projects/OFFSET"
FASHIONIQ_ROOT="/DATA/home/ljq/Projects/OFFSET/data/fashionIQ_dataset/"  # must end with '/'

# Training configuration (you can adjust values below as needed)
LOCAL_RANK="-1"
FASHIONIQ_SPLIT="val-split"
SHOES_PATH=""
CIRR_PATH=""

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
I_TAG="0"
NO_AMP="--no_amp"  # set to empty string to enable AMP

run_train() {
  local CATEGORY="$1"  # one of: dress, shirt, toptee
  echo "[INFO] Starting training for category: ${CATEGORY}"

  python -u "${PROJECT_ROOT}/train.py" \
    --local_rank "${LOCAL_RANK}" \
    --dataset "${CATEGORY}" \
    --fashioniq_split "${FASHIONIQ_SPLIT}" \
    --fashioniq_path "${FASHIONIQ_ROOT}" \
    --shoes_path "${SHOES_PATH}" \
    --cirr_path "${CIRR_PATH}" \
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
}

# Train all three categories sequentially. Comment out lines to skip categories.
run_train $CATEGORY
# run_train "dress"
# run_train "shirt"
# run_train "toptee"

echo "[DONE] All categories finished. Check outputs under ${MODEL_DIR}"