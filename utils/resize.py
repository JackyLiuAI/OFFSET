import argparse
import json
import os
from typing import Dict, List, Set, Optional

from PIL import Image
from joblib import Parallel, delayed
import multiprocessing


def load_json_safely(path: str) -> Optional[object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def collect_ids_from_obj(obj, acc: Set[str]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("candidate", "target") and isinstance(v, str):
                acc.add(v)
            else:
                collect_ids_from_obj(v, acc)
    elif isinstance(obj, list):
        for it in obj:
            collect_ids_from_obj(it, acc)


def collect_category_ids(captions_dir: str, category: str) -> Set[str]:
    ids: Set[str] = set()
    for split in ["train", "val", "test"]:
        path = os.path.join(captions_dir, f"cap.{category}.{split}.json")
        data = load_json_safely(path)
        if data is None:
            continue
        collect_ids_from_obj(data, ids)
    return ids


def find_image_path(images_dir: str, image_id: str) -> Optional[str]:
    # Prefer common extensions
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(images_dir, image_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def resize_image(img: Image.Image, size: int) -> Image.Image:
    # Always convert to RGB to avoid palette issues
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def save_resized(image_path: str, output_path: str, size: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img = resize_image(img, size)
        # Save consistently as JPEG to match typical dataset expectations
        img.save(output_path, format="JPEG")


def process_category(
    category: str,
    ids: Set[str],
    images_dir: str,
    output_root: str,
    size: int,
    i_offset: int = 0,
):
    out_dir = os.path.join(output_root, category)
    os.makedirs(out_dir, exist_ok=True)
    missing: List[str] = []

    id_list = sorted(list(ids))
    num_images = len(id_list)
    num_cores = multiprocessing.cpu_count()
    print(f"[INFO] {category}: {num_images} ids, resizing on {num_cores} CPUs")

    def op(idx: int, image_id: str):
        in_path = find_image_path(images_dir, image_id)
        if in_path is None:
            missing.append(image_id)
            return
        out_path = os.path.join(out_dir, image_id + ".jpg")
        # skip if exists unless overwrite requested
        if (not args.overwrite) and os.path.exists(out_path):
            return
        save_resized(in_path, out_path, size)
        if (idx + 1) % 200 == 0:
            print(f"[{idx+1}/{num_images}] {category} resized")

    Parallel(n_jobs=num_cores)(delayed(op)(i, image_id) for i, image_id in enumerate(id_list))

    if missing:
        miss_file = os.path.join(output_root, f"missing_{category}.txt")
        with open(miss_file, "w", encoding="utf-8") as f:
            for mid in missing:
                f.write(mid + "\n")
        print(f"[WARN] {category}: missing {len(missing)} images, list saved to {miss_file}")
    else:
        print(f"[DONE] {category}: all {num_images} images resized -> {out_dir}")


def collect_cirr_image_paths(root: str) -> List[str]:
    split_dir = os.path.join(root, "image_splits")
    paths: Set[str] = set()
    for split in ["train", "val", "test1"]:
        fname = f"split.rc2.{split}.json"
        spath = os.path.join(split_dir, fname)
        data = load_json_safely(spath)
        if not isinstance(data, dict):
            print(f"[WARN] Bad or missing split file: {spath}")
            continue
        for _, rel in data.items():
            if isinstance(rel, str):
                # skip already generated masks
                if rel.endswith("-segmask.png"):
                    continue
                paths.add(rel)
    return sorted(list(paths))


def resize_cirr_images(root: str, output_dir: str, size: int):
    image_paths = collect_cirr_image_paths(root)
    num_images = len(image_paths)
    if num_images == 0:
        print("[WARN] CIRR: no image paths collected from split files")
        return
    num_cores = multiprocessing.cpu_count()
    print(f"[INFO] CIRR: {num_images} images, resizing on {num_cores} CPUs")

    def op(idx: int, rel_path: str):
        in_path = os.path.join(root, rel_path.lstrip("./"))
        out_path = os.path.join(output_dir, rel_path.lstrip("./"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # skip if exists unless overwrite requested
        if (not args.overwrite) and os.path.exists(out_path):
            return
        with open(in_path, "rb") as f:
            img = Image.open(f)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            # Preserve PNG for CIRR
            img.save(out_path, format="PNG")
        if (idx + 1) % 200 == 0:
            print(f"[{idx+1}/{num_images}] CIRR resized")

    Parallel(n_jobs=num_cores)(delayed(op)(i, p) for i, p in enumerate(image_paths))
    print(f"[DONE] CIRR: resized {num_images} images -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Resize dataset images")
    parser.add_argument("--dataset", choices=["fashioniq", "cirr"], default="fashioniq")
    # FashionIQ args
    parser.add_argument("--images_dir", type=str, default="data/fashionIQ_dataset/images")
    parser.add_argument("--captions_dir", type=str, default="data/fashionIQ_dataset/captions")
    parser.add_argument("--output_dir", type=str, default="data/fashionIQ_dataset/resized_image")
    # CIRR args
    parser.add_argument("--root", type=str, default="data/CIRR")
    # Common args
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--categories", nargs="*", default=["dress", "shirt", "toptee"])
    parser.add_argument("--overwrite", action="store_true")

    global args
    args = parser.parse_args()

    size = args.image_size

    if args.dataset == "fashioniq":
        images_dir = args.images_dir.rstrip("/")
        captions_dir = args.captions_dir.rstrip("/")
        output_dir = args.output_dir.rstrip("/")
        if not os.path.isdir(images_dir):
            raise SystemExit(f"Images directory not found: {images_dir}")
        if not os.path.isdir(captions_dir):
            raise SystemExit(f"Captions directory not found: {captions_dir}")
        os.makedirs(output_dir, exist_ok=True)
        cat_to_ids: Dict[str, Set[str]] = {}
        for cat in args.categories:
            ids = collect_category_ids(captions_dir, cat)
            if not ids:
                print(f"[WARN] No IDs collected for {cat}. Check caption files.")
            cat_to_ids[cat] = ids
        for cat, ids in cat_to_ids.items():
            if not ids:
                continue
            process_category(cat, ids, images_dir, output_dir, size)
    else:  # CIRR
        root = args.root.rstrip("/")
        output_dir = args.output_dir.rstrip("/")
        if not os.path.isdir(root):
            raise SystemExit(f"CIRR root not found: {root}")
        os.makedirs(output_dir, exist_ok=True)
        resize_cirr_images(root, output_dir, size)


if __name__ == "__main__":
    main()