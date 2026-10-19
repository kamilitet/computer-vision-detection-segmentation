"""Quick environment check."""

import torch
from ultralytics import YOLO  # noqa: F401

print("Ultralytics import: OK")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
