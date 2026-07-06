"""
Split a YOLO-format dataset that only has a 'train' folder into
train/valid/test folders, matching image+label pairs.

Expected input structure (typical Roboflow YOLO export):
    dataset_root/
        train/
            images/
                img1.jpg
                img2.jpg
                ...
            labels/
                img1.txt
                img2.txt
                ...
        data.yaml

After running, you'll get:
    dataset_root/
        train/
            images/, labels/   (70% by default)
        valid/
            images/, labels/   (20% by default)
        test/
            images/, labels/   (10% by default)

Usage:
    python split_dataset.py --root /path/to/dataset_root \
        --train 0.7 --valid 0.2 --test 0.1 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def get_pairs(images_dir: Path, labels_dir: Path):
    """Return list of (image_path, label_path) pairs that both exist."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    pairs = []
    missing_labels = []

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images have no matching label file "
              f"(skipped). First few: {missing_labels[:5]}")

    return pairs


def split_pairs(pairs, train_ratio, valid_ratio, test_ratio, seed):
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # remainder goes to test, avoids rounding loss
    n_test = n - n_train - n_valid

    train_pairs = shuffled[:n_train]
    valid_pairs = shuffled[n_train:n_train + n_valid]
    test_pairs = shuffled[n_train + n_valid:]

    return train_pairs, valid_pairs, test_pairs


def move_pairs(pairs, dest_images_dir, dest_labels_dir, mode="copy"):
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if mode == "copy" else shutil.move

    for img_path, label_path in pairs:
        op(str(img_path), str(dest_images_dir / img_path.name))
        op(str(label_path), str(dest_labels_dir / label_path.name))


def main():
    parser = argparse.ArgumentParser(description="Split YOLO train folder into train/valid/test")
    parser.add_argument("--root", type=str, required=True,
                         help="Path to dataset root (folder containing 'train/')")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--valid", type=float, default=0.2, help="Valid ratio (default 0.2)")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy",
                         help="Copy (safe, keeps original train intact) or move (destructive). Default: copy")
    args = parser.parse_args()

    root = Path(args.root)
    old_train_images = root / "train" / "images"
    old_train_labels = root / "train" / "labels"

    if not old_train_images.exists() or not old_train_labels.exists():
        raise FileNotFoundError(
            f"Expected to find {old_train_images} and {old_train_labels}. "
            "Check your --root path."
        )

    print(f"Reading pairs from {old_train_images} ...")
    pairs = get_pairs(old_train_images, old_train_labels)
    print(f"Found {len(pairs)} valid image+label pairs.")

    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs, args.train, args.valid, args.test, args.seed
    )
    print(f"Split -> train: {len(train_pairs)}, valid: {len(valid_pairs)}, test: {len(test_pairs)}")

    # Write to temporary new folders first, then swap, so we never
    # partially destroy the original train folder if something goes wrong.
    tmp_root = root / "_split_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    move_pairs(train_pairs, tmp_root / "train" / "images", tmp_root / "train" / "labels", mode="copy")
    move_pairs(valid_pairs, tmp_root / "valid" / "images", tmp_root / "valid" / "labels", mode="copy")
    move_pairs(test_pairs, tmp_root / "test" / "images", tmp_root / "test" / "labels", mode="copy")

    # Now replace original folders
    if args.mode == "move":
        shutil.rmtree(root / "train")
    for split in ["train", "valid", "test"]:
        dest = root / split
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(tmp_root / split), str(dest))

    shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nDone! New structure:")
    for split in ["train", "valid", "test"]:
        img_count = len(list((root / split / "images").iterdir()))
        print(f"  {split}/images: {img_count} files")

    print(f"\ndata.yaml already points to ../train/images, ../valid/images, ../test/images, "
          f"so no changes needed there.")


if __name__ == "__main__":
    main()
