#!/usr/bin/env bash
# set -euo pipefail

# One-click preprocessing for FashionIQ: resize images and generate CLIPSeg masks.

# Paths
PROJECT_ROOT="/DATA/home/ljq/Projects/OFFSET"
FASHIONIQ_ROOT="/DATA/home/ljq/Projects/OFFSET/data/fashionIQ_dataset"  # no trailing slash

# Config
IMAGE_SIZE=256
CATEGORIES=(dress shirt toptee)
OVERWRITE=1                 # set to 0 to skip existing files
SAVE_SEG_IMAGE=0            # set to 1 to also save segmented images
GPU_ID=""                   # e.g., "0" to pin to a specific GPU, empty to use default

echo "[INFO] FashionIQ preprocess: root=${FASHIONIQ_ROOT}, size=${IMAGE_SIZE}, categories=${CATEGORIES[*]}"

# Resize images into <root>/resized_image/<category>/id.jpg
echo "[STEP] Resizing FashionIQ images..."
if [[ -n "${GPU_ID}" ]]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
python -u "${PROJECT_ROOT}/utils/resize.py" \
  --dataset fashioniq \
  --images_dir "${FASHIONIQ_ROOT}/images" \
  --captions_dir "${FASHIONIQ_ROOT}/captions" \
  --output_dir "${FASHIONIQ_ROOT}/resized_image" \
  --image_size ${IMAGE_SIZE} \
  --categories "${CATEGORIES[@]}" \
  $( ((OVERWRITE)) && echo "--overwrite" )

# Generate masks into <root>/resized_image/<category>_segmask/id-seg.png
echo "[STEP] Generating CLIPSeg masks for FashionIQ..."
if [[ -n "${GPU_ID}" ]]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
python -u "${PROJECT_ROOT}/utils/generate_segmentation.py" \
  --dataset fashioniq \
  --root "${FASHIONIQ_ROOT}/" \
  --categories "${CATEGORIES[@]}" \
  $( ((OVERWRITE)) && echo "--overwrite" ) \
  $( ((SAVE_SEG_IMAGE)) && echo "--save-seg-image" )

echo "[DONE] FashionIQ preprocess complete. Outputs under ${FASHIONIQ_ROOT}/resized_image"