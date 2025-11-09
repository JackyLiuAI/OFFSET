#!/usr/bin/env python3
"""
Generate segmentation masks using CLIPSeg and save them to disk
for datasets supported by this repo.

Supported datasets and output conventions:
- FashionIQ:
  Images: <root>/resized_image/<category>/*.jpg
  Masks:  <root>/resized_image/<category>_segmask/<id>-seg.png

- Shoes:
  Images: <root>/**/*.jpg (recursive)
  Masks:  same directory, filename: <name>-segmask.jpg

- CIRR:
  Images: <root>/**/*.png (recursive)
  Masks:  same directory, filename: <name>-segmask.png

Usage examples:
  FashionIQ (all categories):
    python scripts/generate_segmentation.py \
      --dataset fashioniq --root data/fashionIQ_dataset \
      --categories dress shirt toptee

  Shoes:
    python scripts/generate_segmentation.py --dataset shoes --root /path/to/shoes

  CIRR:
    python scripts/generate_segmentation.py --dataset cirr --root /path/to/cirr

Requires: transformers, torch, Pillow
"""

import os
import glob
import argparse
from typing import Optional

import torch
from PIL import Image

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'transformers'. Please install with: pip install transformers"
    ) from e


def init_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_clipseg(device: str):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()
    return processor, model


def segment_and_save(
    image_path: str,
    text: str,
    out_mask_path: str,
    processor: CLIPSegProcessor,
    model: CLIPSegForImageSegmentation,
    device: str,
    out_seg_image_path: Optional[str] = None,
    overwrite: bool = False,
):
    # Skip if exists
    if not overwrite and os.path.exists(out_mask_path):
        return False

    # Load image robustly
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")

    # Prepare inputs
    inputs = processor(text=[text], images=[img], padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # logits -> probability mask
    logits = outputs.logits  # [B, H, W]
    mask = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [H, W], 0..1
    mask_img = Image.fromarray((mask * 255).astype("uint8"))

    # Save mask
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    mask_img.save(out_mask_path)

    # Optionally save segmented image (processed image * mask)
    if out_seg_image_path:
        # inputs.pixel_values: [B, C, H, W], normalized to 0..1 according to processor
        feat = inputs["pixel_values"][0].permute(1, 2, 0).cpu().numpy() * mask[..., None]
        # Normalize to 0..255
        mn, mx = float(feat.min()), float(feat.max())
        feat = (feat - mn) / (mx - mn + 1e-8)
        seg_img = Image.fromarray((feat * 255).astype("uint8"))
        os.makedirs(os.path.dirname(out_seg_image_path), exist_ok=True)
        seg_img.save(out_seg_image_path)

    return True


def generate_fashioniq_masks(root: str, categories: list[str], prompt: Optional[str], overwrite: bool, save_seg_image: bool = False):
    device = init_device()
    processor, model = init_clipseg(device)

    for cat in categories:
        image_dir = os.path.join(root, "resized_image", cat)
        seg_dir = os.path.join(root, "resized_image", f"{cat}_segmask")
        if not os.path.isdir(image_dir):
            print(f"[WARN] Image directory not found: {image_dir}")
            continue

        img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if not img_paths:
            print(f"[WARN] No images in: {image_dir}")
            continue

        print(f"[INFO] Category '{cat}': {len(img_paths)} images")
        count = 0
        for img_path in img_paths:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            out_mask = os.path.join(seg_dir, f"{img_id}-seg.png")
            text = prompt or cat
            out_seg_image = os.path.join(seg_dir, f"{img_id}-segimg.png") if save_seg_image else None
            ok = segment_and_save(
                img_path, text, out_mask, processor, model, device,
                out_seg_image_path=out_seg_image, overwrite=overwrite
            )
            count += int(ok)
        print(f"[DONE] {cat}: generated {count} masks -> {seg_dir}")


def generate_shoes_masks(root: str, prompt: Optional[str], overwrite: bool):
    device = init_device()
    processor, model = init_clipseg(device)

    img_paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
    print(f"[INFO] Shoes: {len(img_paths)} images")
    count = 0
    for img_path in img_paths:
        out_mask = img_path.replace(".jpg", "-segmask.jpg")
        text = prompt or "shoes"
        ok = segment_and_save(img_path, text, out_mask, processor, model, device, overwrite=overwrite)
        count += int(ok)
    print(f"[DONE] Shoes: generated {count} masks (stored alongside images)")


def generate_cirr_masks(root: str, prompt: Optional[str], overwrite: bool):
    device = init_device()
    processor, model = init_clipseg(device)
    # Collect image paths from split files to avoid processing masks and unrelated files
    split_dir = os.path.join(root, "image_splits")
    img_paths = []
    for split in ["train", "val", "test1"]:
        spath = os.path.join(split_dir, f"split.rc2.{split}.json")
        if not os.path.exists(spath):
            continue
        try:
            data = torch.jit.load(spath)  # intentionally wrong to fall back
        except Exception:
            # normal JSON load
            import json
            with open(spath, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for _, rel in mapping.items():
                if isinstance(rel, str) and rel.endswith('.png') and not rel.endswith('-segmask.png'):
                    img_paths.append(os.path.join(root, rel.lstrip('./')))

    img_paths = sorted(list(set(img_paths)))
    print(f"[INFO] CIRR: {len(img_paths)} images from split files")
    count = 0
    for img_path in img_paths:
        out_mask = img_path.replace(".png", "-segmask.png")
        text = prompt or "dress"
        ok = segment_and_save(img_path, text, out_mask, processor, model, device, overwrite=overwrite)
        count += int(ok)
    print(f"[DONE] CIRR: generated {count} masks (stored alongside images)")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate segmentation masks with CLIPSeg")
    parser.add_argument("--dataset", choices=["fashioniq", "shoes", "cirr"],
                        help="Dataset type", default="fashioniq")
    parser.add_argument("--root", type=str,
                        help="Dataset root directory",
                        default="/DATA/home/ljq/Projects/OFFSET/data/fashionIQ_dataset/")
    parser.add_argument("--categories", nargs="*", default=["dress", "shirt", "toptee"],
                        help="FashionIQ categories (only used for --dataset fashioniq)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for segmentation (default: dataset/category name)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing masks")
    parser.add_argument("--save-seg-image", action="store_true",
                        help="Also save segmented image next to masks (FashionIQ only)")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.rstrip("/")
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory not found: {root}")

    if args.dataset == "fashioniq":
        generate_fashioniq_masks(root, args.categories, args.prompt, args.overwrite, save_seg_image=args.save_seg_image)
    elif args.dataset == "shoes":
        generate_shoes_masks(root, args.prompt, args.overwrite)
    elif args.dataset == "cirr":
        generate_cirr_masks(root, args.prompt, args.overwrite)
    else:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()