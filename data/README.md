# Data

This repository does **not** include the raw PASCAL VOC dataset or generated training subsets.

The scripts expect PASCAL VOC 2012 to be extracted to:

```text
data/raw/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/
├── Annotations/
├── JPEGImages/
└── SegmentationClass/
```

Then prepare the two tasks with:

```bash
python scripts/prepare_voc_detection.py
python scripts/prepare_voc_segmentation.py
```

Generated data are written to `data/processed_detection/` and `data/processed_segmentation/` and are ignored by Git.

For the vehicle-tracking demo, place a local video at `data/video/input.mp4` or pass a custom path with `--input`.
