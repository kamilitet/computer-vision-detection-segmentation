"""Train YOLOv8n either from scratch or from pretrained weights."""

import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--data", default="data/processed_detection/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detection")
    args = parser.parse_args()

    model_source = "yolov8n.yaml" if args.mode == "scratch" else "yolov8n.pt"
    epochs = args.epochs if args.epochs is not None else (50 if args.mode == "scratch" else 30)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    model = YOLO(model_source)
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.mode,
        device=device,
        workers=0,
    )


if __name__ == "__main__":
    main()
