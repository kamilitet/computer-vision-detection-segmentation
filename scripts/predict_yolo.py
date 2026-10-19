"""Run YOLO predictions using a trained model checkpoint."""

import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="data/processed_detection/images/val")
    parser.add_argument("--output_name", default="predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        save=True,
        project="runs/detection/predictions",
        name=args.output_name,
        conf=args.conf,
        device=device,
    )


if __name__ == "__main__":
    main()
