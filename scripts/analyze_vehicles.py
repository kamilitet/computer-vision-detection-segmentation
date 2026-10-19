"""Track vehicles in a video and classify track histories as moving/standing."""

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck


def center_of_box(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def distance(point_a, point_b):
    return math.dist(point_a, point_b)


def movement_state(track_history, threshold_px):
    if len(track_history) < 2:
        return "new"
    return "moving" if distance(track_history[0], track_history[-1]) >= threshold_px else "standing"


def draw_label(frame, text, x, y, color):
    cv2.rectangle(frame, (x, y - 25), (x + 220, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/video/input.mp4")
    parser.add_argument("--output_video", default="runs/video/vehicle_analysis.mp4")
    parser.add_argument("--output_csv", default="runs/video/vehicle_analysis_summary.csv")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--movement_threshold_px", type=float, default=20.0)
    parser.add_argument("--history_seconds", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.4)
    args = parser.parse_args()

    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    history_length = max(int(fps * args.history_seconds), 5)
    track_history = defaultdict(lambda: deque(maxlen=history_length))
    track_states = defaultdict(set)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            conf=args.confidence,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                track_history[track_id].append(center_of_box((x1, y1, x2, y2)))
                state = movement_state(track_history[track_id], args.movement_threshold_px)
                if state != "new":
                    track_states[track_id].add(state)

                if {"standing", "moving"}.issubset(track_states[track_id]):
                    color, label = (0, 255, 255), "standing + moving"
                elif "moving" in track_states[track_id]:
                    color, label = (0, 0, 255), "moving"
                elif "standing" in track_states[track_id]:
                    color, label = (0, 255, 0), "standing"
                else:
                    color, label = (255, 0, 0), "new"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"ID {track_id}: {label}", x1, max(y1, 30), color)

        writer.write(frame)

    capture.release()
    writer.release()

    counts = {"standing": 0, "moving": 0, "standing_and_moving": 0}
    for states in track_states.values():
        if {"standing", "moving"}.issubset(states):
            counts["standing_and_moving"] += 1
        elif "moving" in states:
            counts["moving"] += 1
        elif "standing" in states:
            counts["standing"] += 1

    with open(args.output_csv, "w", newline="", encoding="utf-8") as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(["category", "count"])
        writer_csv.writerows(counts.items())

    print(counts)
    print("Note: movement is a pixel-displacement heuristic, not a physical speed measurement.")


if __name__ == "__main__":
    main()
