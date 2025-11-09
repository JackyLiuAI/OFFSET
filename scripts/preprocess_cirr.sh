#!/usr/bin/env bash
# set -euo pipefail

# One-click preprocessing for CIRR: resize images and generate CLIPSeg masks.

# Paths
PROJECT_ROOT="/DATA/home/ljq/Projects/OFFSET"
CIRR_ROOT="/DATA/home/ljq/Projects/OFFSET/data/CIRR"  # no trailing slash

# Config
IMAGE_SIZE=256
OVERWRITE=1               # set to 0 to skip existing files
GPU_ID=""                 # e.g., "0" to pin to a specific GPU, empty to use default

echo "[INFO] CIRR preprocess: root=${CIRR_ROOT}, size=${IMAGE_SIZE}"

# Resize images using split files into <output_dir>/<relative_path>.png
echo "[STEP] Resizing CIRR images..."
if [[ -n "${GPU_ID}" ]]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
python -u "${PROJECT_ROOT}/utils/resize.py" \
  --dataset cirr \
  --root "${CIRR_ROOT}" \
  --output_dir "${CIRR_ROOT}/resized_image" \
  --image_size ${IMAGE_SIZE} \
  $( ((OVERWRITE)) && echo "--overwrite" )

# Generate masks alongside images: <path>-segmask.png
echo "[STEP] Generating CLIPSeg masks for CIRR..."
if [[ -n "${GPU_ID}" ]]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
python -u "${PROJECT_ROOT}/utils/generate_segmentation.py" \
  --dataset cirr \
  --root "${CIRR_ROOT}" \
  $( ((OVERWRITE)) && echo "--overwrite" )

echo "[DONE] CIRR preprocess complete. Resized under ${CIRR_ROOT}/resized_image; masks next to originals"