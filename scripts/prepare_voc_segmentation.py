"""Prepare a 3-class PASCAL VOC 2012 semantic-segmentation subset."""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

VOC_CLASS_IDS = {"cow": 10, "motorbike": 14}
NEW_CLASS_IDS = {"background": 0, "cow": 1, "motorbike": 2}
IGNORE_INDEX = 255


def convert_mask(original_mask: Image.Image) -> Image.Image:
    mask = np.array(original_mask)
    converted = np.zeros_like(mask, dtype=np.uint8)
    converted[mask == IGNORE_INDEX] = IGNORE_INDEX
    converted[mask == VOC_CLASS_IDS["cow"]] = NEW_CLASS_IDS["cow"]
    converted[mask == VOC_CLASS_IDS["motorbike"]] = NEW_CLASS_IDS["motorbike"]
    return Image.fromarray(converted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voc_root",
        type=Path,
        default=Path("data/raw/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed_segmentation"))
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = args.voc_root / "JPEGImages"
    mask_dir = args.voc_root / "SegmentationClass"

    image_train = args.output / "images/train"
    image_val = args.output / "images/val"
    mask_train = args.output / "masks/train"
    mask_val = args.output / "masks/val"
    for directory in [image_train, image_val, mask_train, mask_val]:
        directory.mkdir(parents=True, exist_ok=True)

    selected_ids = []
    for mask_path in mask_dir.glob("*.png"):
        mask = np.array(Image.open(mask_path))
        if np.any(mask == VOC_CLASS_IDS["cow"]) or np.any(mask == VOC_CLASS_IDS["motorbike"]):
            selected_ids.append(mask_path.stem)

    rng = random.Random(args.seed)
    rng.shuffle(selected_ids)
    split_index = int(len(selected_ids) * args.train_fraction)
    train_ids = selected_ids[:split_index]
    val_ids = selected_ids[split_index:]

    def save_split(ids, image_out_dir, mask_out_dir):
        for image_id in ids:
            source_image = image_dir / f"{image_id}.jpg"
            source_mask = mask_dir / f"{image_id}.png"
            if not source_image.exists() or not source_mask.exists():
                continue
            shutil.copy2(source_image, image_out_dir / source_image.name)
            convert_mask(Image.open(source_mask)).save(mask_out_dir / source_mask.name)

    save_split(train_ids, image_train, mask_train)
    save_split(val_ids, image_val, mask_val)

    print(f"Selected masks: {len(selected_ids)}")
    print(f"Train: {len(train_ids)} | Validation: {len(val_ids)}")
    print(f"Dataset written to: {args.output}")


if __name__ == "__main__":
    main()
