#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from glob import glob


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str, overwrite: bool = False):
    if not os.path.exists(src):
        print(f"[WARN] Missing source: {src}")
        return
    if os.path.exists(dst) and not overwrite:
        return
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def filter_json_items(src_json_path: str, dst_json_path: str, limit: int, overwrite: bool):
    with open(src_json_path, 'r') as f:
        items = json.load(f)
    subset = items[:limit]
    ensure_dir(os.path.dirname(dst_json_path))
    if os.path.exists(dst_json_path) and not overwrite:
        print(f"[INFO] JSON exists (skip): {dst_json_path}")
    else:
        with open(dst_json_path, 'w') as f:
            json.dump(subset, f)
        print(f"[OK] Wrote subset {len(subset)} -> {dst_json_path}")
    return subset


def filter_split_list(src_split_path: str, dst_split_path: str, allowed_ids: set, overwrite: bool):
    # Split files are lists of image IDs
    if not os.path.exists(src_split_path):
        print(f"[WARN] Split not found: {src_split_path}")
        return
    with open(src_split_path, 'r') as f:
        images = json.load(f)
    filtered = [iid for iid in images if iid in allowed_ids]
    ensure_dir(os.path.dirname(dst_split_path))
    if os.path.exists(dst_split_path) and not overwrite:
        print(f"[INFO] Split exists (skip): {dst_split_path}")
    else:
        with open(dst_split_path, 'w') as f:
            json.dump(filtered, f)
        print(f"[OK] Wrote split {len(filtered)} -> {dst_split_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract FashionIQ subset (N per JSON) with images and masks.")
    parser.add_argument('--src', default='data/fashionIQ_dataset/', help='Source FashionIQ root (must end with /)')
    parser.add_argument('--dst', default='data/fashionIQ_test/', help='Destination root for subset (will be created)')
    parser.add_argument('--limit', type=int, default=10, help='Number of items to keep per caption JSON')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files in destination')
    args = parser.parse_args()

    src_root = args.src
    dst_root = args.dst

    captions_src = os.path.join(src_root, 'captions')
    splits_src = os.path.join(src_root, 'image_splits')
    images_src = os.path.join(src_root, 'resized_image')

    captions_dst = os.path.join(dst_root, 'captions')
    splits_dst = os.path.join(dst_root, 'image_splits')
    images_dst = os.path.join(dst_root, 'resized_image')

    ensure_dir(captions_dst)
    ensure_dir(splits_dst)
    ensure_dir(images_dst)

    # Find caption JSONs: cap.<category>.<split>.json
    caption_files = sorted(glob(os.path.join(captions_src, 'cap.*.*.json')))
    if not caption_files:
        print(f"[ERROR] No caption JSONs found under {captions_src}")
        return

    for cap_path in caption_files:
        cap_name = os.path.basename(cap_path)  # cap.dress.train.json
        parts = cap_name.split('.')  # ['cap', '<category>', '<split>', 'json']
        if len(parts) != 4 or parts[0] != 'cap':
            print(f"[WARN] Skip non-standard caption file: {cap_name}")
            continue
        category = parts[1]
        split = parts[2]
        dst_cap_path = os.path.join(captions_dst, cap_name)

        # Filter caption JSON
        subset = filter_json_items(cap_path, dst_cap_path, args.limit, args.overwrite)

        # Build set of image ids used by this subset
        used_ids = set()
        for item in subset:
            cand = item.get('candidate')
            targ = item.get('target')
            if cand:
                used_ids.add(cand)
            if targ:
                used_ids.add(targ)

        # Filter split JSON if available (train/val both handled)
        split_src_path = os.path.join(splits_src, f'split.{category}.{split}.json')
        split_dst_path = os.path.join(splits_dst, f'split.{category}.{split}.json')
        filter_split_list(split_src_path, split_dst_path, used_ids, args.overwrite)

        # Copy images and masks
        img_cat_src = os.path.join(images_src, category)
        img_cat_dst = os.path.join(images_dst, category)
        seg_cat_src = os.path.join(images_src, f'{category}_segmask')
        seg_cat_dst = os.path.join(images_dst, f'{category}_segmask')
        ensure_dir(img_cat_dst)
        ensure_dir(seg_cat_dst)

        copied_img, copied_seg = 0, 0
        for iid in sorted(used_ids):
            img_src = os.path.join(img_cat_src, f'{iid}.jpg')
            img_dst = os.path.join(img_cat_dst, f'{iid}.jpg')
            seg_src = os.path.join(seg_cat_src, f'{iid}-seg.png')
            seg_dst = os.path.join(seg_cat_dst, f'{iid}-seg.png')
            copy_file(img_src, img_dst, args.overwrite)
            copy_file(seg_src, seg_dst, args.overwrite)
            if os.path.exists(img_dst):
                copied_img += 1
            if os.path.exists(seg_dst):
                copied_seg += 1

        print(f"[OK] {category}.{split}: copied {copied_img} images, {copied_seg} masks")

    # Optional: copy correction dicts if present
    for category in ['dress', 'shirt', 'toptee']:
        corr_src = os.path.join(captions_src, f'correction_dict_{category}.json')
        corr_dst = os.path.join(captions_dst, f'correction_dict_{category}.json')
        if os.path.exists(corr_src):
            copy_file(corr_src, corr_dst, args.overwrite)
            print(f"[OK] Copied correction_dict for {category}")

    print(f"[DONE] Subset extraction completed -> {dst_root}")


if __name__ == '__main__':
    main()