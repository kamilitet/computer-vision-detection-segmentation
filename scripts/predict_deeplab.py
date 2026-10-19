"""Visualize DeepLabV3 semantic-segmentation predictions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

NUM_CLASSES = 3


def create_model():
    try:
        return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
    except TypeError:
        return deeplabv3_resnet50(pretrained=False, num_classes=NUM_CLASSES)


def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 0] = [0, 0, 0]
    color_mask[mask == 1] = [0, 255, 0]
    color_mask[mask == 2] = [255, 0, 0]
    return color_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source", default="data/processed_segmentation/images/val")
    parser.add_argument("--output", default="runs/segmentation/predictions")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_paths = sorted(Path(args.source).glob("*.jpg"))[: args.limit]
    with torch.no_grad():
        for image_path in image_paths:
            original = Image.open(image_path).convert("RGB")
            resized = original.resize((args.image_size, args.image_size))
            tensor = transform(original).unsqueeze(0).to(device)
            prediction = torch.argmax(model(tensor)["out"], dim=1).squeeze().cpu().numpy()
            color_mask = colorize_mask(prediction)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(resized)
            axes[0].set_title("Original image")
            axes[0].axis("off")
            axes[1].imshow(resized)
            axes[1].imshow(color_mask, alpha=0.5)
            axes[1].set_title("Predicted segmentation")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / f"{image_path.stem}_prediction.png", bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
