#!/usr/bin/env python3
import argparse
import json
import os
import shutil


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str, overwrite: bool = False):
    if not os.path.exists(src):
        print(f"[WARN] Missing source: {src}")
        return False
    if os.path.exists(dst) and not overwrite:
        return True
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(obj, path: str, overwrite: bool):
    ensure_dir(os.path.dirname(path))
    if os.path.exists(path) and not overwrite:
        print(f"[INFO] JSON exists (skip): {path}")
        return
    with open(path, 'w') as f:
        json.dump(obj, f)
    print(f"[OK] Wrote {path}")


def filter_captions(src_path: str, dst_path: str, limit: int, overwrite: bool):
    items = load_json(src_path)
    subset = items[:limit]
    save_json(subset, dst_path, overwrite)
    return subset


def filter_split_dict(src_path: str, dst_path: str, allowed_names: set, overwrite: bool):
    if not os.path.exists(src_path):
        print(f"[WARN] Split not found: {src_path}")
        return {}
    mapping = load_json(src_path)
    filtered = {k: v for k, v in mapping.items() if k in allowed_names}
    save_json(filtered, dst_path, overwrite)
    return filtered


def copy_images_and_masks(src_root: str, dst_root: str, name_to_path: dict, overwrite: bool):
    copied_img, copied_seg = 0, 0
    for name, rel_path in name_to_path.items():
        rel = rel_path.lstrip('./')
        src_img = os.path.join(src_root, rel)
        dst_img = os.path.join(dst_root, rel)
        if copy_file(src_img, dst_img, overwrite):
            copied_img += 1
        # mask: same directory, replace .png with -segmask.png
        src_mask = src_img.replace('.png', '-segmask.png')
        dst_mask = dst_img.replace('.png', '-segmask.png')
        if copy_file(src_mask, dst_mask, overwrite):
            copied_seg += 1
    print(f"[OK] Copied {copied_img} images, {copied_seg} masks")


def main():
    parser = argparse.ArgumentParser(description="Extract CIRR subset (N per caption JSON) and copy images/masks.")
    parser.add_argument('--src', default='data/CIRR/', help='Source CIRR root (must end with /)')
    parser.add_argument('--dst', default='data/CIRR_test/', help='Destination root for subset (will be created)')
    parser.add_argument('--limit', type=int, default=10, help='Number of items to keep per caption JSON')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files in destination')
    args = parser.parse_args()

    src_root = args.src
    dst_root = args.dst

    captions_src = os.path.join(src_root, 'captions')
    splits_src = os.path.join(src_root, 'image_splits')
    captions_dst = os.path.join(dst_root, 'captions')
    splits_dst = os.path.join(dst_root, 'image_splits')

    ensure_dir(captions_dst)
    ensure_dir(splits_dst)

    # Process train
    train_caps_src = os.path.join(captions_src, 'cap.rc2.train.json')
    train_caps_dst = os.path.join(captions_dst, 'cap.rc2.train.json')
    if os.path.exists(train_caps_src):
        train_subset = filter_captions(train_caps_src, train_caps_dst, args.limit, args.overwrite)
        train_names = set()
        for it in train_subset:
            if 'reference' in it: train_names.add(it['reference'])
            if 'target_hard' in it: train_names.add(it['target_hard'])
        train_split_src = os.path.join(splits_src, 'split.rc2.train.json')
        train_split_dst = os.path.join(splits_dst, 'split.rc2.train.json')
        train_map = filter_split_dict(train_split_src, train_split_dst, train_names, args.overwrite)
        copy_images_and_masks(src_root, dst_root, train_map, args.overwrite)
    else:
        print(f"[WARN] Missing {train_caps_src}, skip train subset")

    # Process val
    val_caps_src = os.path.join(captions_src, 'cap.rc2.val.json')
    val_caps_dst = os.path.join(captions_dst, 'cap.rc2.val.json')
    if os.path.exists(val_caps_src):
        val_subset = filter_captions(val_caps_src, val_caps_dst, args.limit, args.overwrite)
        val_names = set()
        for it in val_subset:
            if 'reference' in it: val_names.add(it['reference'])
            if 'target_hard' in it: val_names.add(it['target_hard'])
            # also include gallery subset members to be safe
            for m in it.get('img_set', {}).get('members', []):
                val_names.add(m)
        val_split_src = os.path.join(splits_src, 'split.rc2.val.json')
        val_split_dst = os.path.join(splits_dst, 'split.rc2.val.json')
        val_map = filter_split_dict(val_split_src, val_split_dst, val_names, args.overwrite)
        copy_images_and_masks(src_root, dst_root, val_map, args.overwrite)
    else:
        print(f"[WARN] Missing {val_caps_src}, skip val subset")

    # Process test1
    test_caps_src = os.path.join(captions_src, 'cap.rc2.test1.json')
    test_caps_dst = os.path.join(captions_dst, 'cap.rc2.test1.json')
    if os.path.exists(test_caps_src):
        test_subset = filter_captions(test_caps_src, test_caps_dst, args.limit, args.overwrite)
        test_names = set()
        for it in test_subset:
            if 'reference' in it: test_names.add(it['reference'])
            for m in it.get('img_set', {}).get('members', []):
                test_names.add(m)
        test_split_src = os.path.join(splits_src, 'split.rc2.test1.json')
        test_split_dst = os.path.join(splits_dst, 'split.rc2.test1.json')
        test_map = filter_split_dict(test_split_src, test_split_dst, test_names, args.overwrite)
        copy_images_and_masks(src_root, dst_root, test_map, args.overwrite)
    else:
        print(f"[WARN] Missing {test_caps_src}, skip test1 subset")

    print(f"[DONE] CIRR subset extraction completed -> {dst_root}")


if __name__ == '__main__':
    main()