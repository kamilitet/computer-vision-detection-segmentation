"""Prepare a cow/motorbike PASCAL VOC 2012 subset in YOLO format."""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_CLASSES = ["cow", "motorbike"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


def convert_box(width: int, height: int, box: tuple[float, float, float, float]):
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voc_root",
        type=Path,
        default=Path("data/raw/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed_detection"))
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations_dir = args.voc_root / "Annotations"
    images_dir = args.voc_root / "JPEGImages"

    images_train = args.output / "images/train"
    images_val = args.output / "images/val"
    labels_train = args.output / "labels/train"
    labels_val = args.output / "labels/val"
    for directory in [images_train, images_val, labels_train, labels_val]:
        directory.mkdir(parents=True, exist_ok=True)

    selected = []
    for xml_file in annotations_dir.glob("*.xml"):
        root = ET.parse(xml_file).getroot()
        filename = root.findtext("filename")
        size = root.find("size")
        if not filename or size is None:
            continue

        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        yolo_lines = []

        for obj in root.findall("object"):
            class_name = obj.findtext("name")
            if class_name not in TARGET_CLASSES:
                continue

            box = obj.find("bndbox")
            if box is None:
                continue
            coords = tuple(float(box.findtext(name)) for name in ("xmin", "ymin", "xmax", "ymax"))
            x_center, y_center, box_width, box_height = convert_box(width, height, coords)
            yolo_lines.append(
                f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        if yolo_lines:
            selected.append((filename, yolo_lines))

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    split_index = int(len(selected) * args.train_fraction)
    train_data = selected[:split_index]
    val_data = selected[split_index:]

    def save_split(data_split, image_out_dir, label_out_dir):
        for filename, labels in data_split:
            source_image = images_dir / filename
            if not source_image.exists():
                continue
            shutil.copy2(source_image, image_out_dir / filename)
            (label_out_dir / f"{Path(filename).stem}.txt").write_text("\n".join(labels), encoding="utf-8")

    save_split(train_data, images_train, labels_train)
    save_split(val_data, images_val, labels_val)

    yaml_text = f"""path: {args.output.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: cow
  1: motorbike
"""
    (args.output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Selected images: {len(selected)}")
    print(f"Train: {len(train_data)} | Validation: {len(val_data)}")
    print(f"Dataset written to: {args.output}")


if __name__ == "__main__":
    main()
