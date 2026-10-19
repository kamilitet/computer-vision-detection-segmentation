"""Train DeepLabV3-ResNet50 for background/cow/motorbike segmentation."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

CLASS_NAMES = ["background", "cow", "motorbike"]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


class VOCSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=256):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        image = self.image_transform(Image.open(image_path).convert("RGB"))
        mask = Image.open(mask_path).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        return image, torch.from_numpy(np.array(mask, dtype=np.int64)).long()


def create_model(mode):
    if mode == "pretrained":
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        except Exception:
            model = deeplabv3_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    else:
        try:
            model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
        except TypeError:
            model = deeplabv3_resnet50(pretrained=False, num_classes=NUM_CLASSES)
    return model


def update_confusion(confusion, predictions, targets):
    valid = targets != IGNORE_INDEX
    predictions = predictions[valid]
    targets = targets[valid]
    encoded = targets * NUM_CLASSES + predictions
    counts = torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES)
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def metrics_from_confusion(confusion):
    confusion = confusion.float()
    intersection = confusion.diag()
    union = confusion.sum(1) + confusion.sum(0) - intersection
    ious = torch.where(union > 0, intersection / union, torch.nan)
    mean_iou = torch.nanmean(ious).item()
    pixel_accuracy = intersection.sum().item() / max(confusion.sum().item(), 1.0)
    return pixel_accuracy, mean_iou, ious.tolist()


def save_plots(history, output_dir):
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(output_dir / "loss_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["epoch"], history["mean_iou"], label="mean_iou")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU")
    plt.legend()
    plt.savefig(output_dir / "mean_iou.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    for class_name in CLASS_NAMES:
        plt.plot(history["epoch"], history[f"iou_{class_name}"], label=class_name)
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU by class")
    plt.legend()
    plt.savefig(output_dir / "class_iou.png", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VOCSegmentationDataset(
        "data/processed_segmentation/images/train",
        "data/processed_segmentation/masks/train",
        args.image_size,
    )
    val_dataset = VOCSegmentationDataset(
        "data/processed_segmentation/images/val",
        "data/processed_segmentation/masks/val",
        args.image_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(args.mode).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output_dir = Path(f"runs/segmentation/deeplab_{args.mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "epoch", "train_loss", "val_loss", "pixel_accuracy", "mean_iou",
        "iou_background", "iou_cow", "iou_motorbike",
    ]
    history = {column: [] for column in columns}
    best_mean_iou = -1.0

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                output = model(images)["out"]
                loss = criterion(output, masks)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            train_loss = train_loss_sum / max(len(train_loader), 1)

            model.eval()
            val_loss_sum = 0.0
            confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    output = model(images)["out"]
                    val_loss_sum += criterion(output, masks).item()
                    update_confusion(confusion, torch.argmax(output, dim=1), masks)

            val_loss = val_loss_sum / max(len(val_loader), 1)
            pixel_accuracy, mean_iou, class_ious = metrics_from_confusion(confusion)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "pixel_accuracy": pixel_accuracy,
                "mean_iou": mean_iou,
                "iou_background": class_ious[0],
                "iou_cow": class_ious[1],
                "iou_motorbike": class_ious[2],
            }
            writer.writerow(row)
            for key, value in row.items():
                history[key].append(value)

            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                torch.save(model.state_dict(), output_dir / "best_model.pth")

            print(
                f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | pixel_acc={pixel_accuracy:.4f} | mIoU={mean_iou:.4f}"
            )

    torch.save(model.state_dict(), output_dir / "last_model.pth")
    save_plots(history, output_dir)
    print(f"Best mean IoU: {best_mean_iou:.4f}")


if __name__ == "__main__":
    main()
