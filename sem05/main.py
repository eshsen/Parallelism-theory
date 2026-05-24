"""
YOLOv8s-pose inference on video file.

Usage:
    python main.py --video_path input.mp4 --mode single --output out_single.mp4
    python main.py --video_path input.mp4 --mode multi  --output out_multi.mp4 --workers 4
"""

import argparse
import time
import queue
import threading
import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# RAII wrappers
# ──────────────────────────────────────────────────────────────────────────────

class VideoReader:
    """RAII wrapper around cv2.VideoCapture for a file."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        return self._cap.read()

    def __del__(self):
        if self._cap.isOpened():
            self._cap.release()


class VideoWriter:
    """RAII wrapper around cv2.VideoWriter."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open output video: {path}")

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def __del__(self):
        if self._writer.isOpened():
            self._writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model: YOLO, frame: np.ndarray) -> np.ndarray:
    """Run YOLOv8s-pose on a single BGR frame and return annotated frame."""
    results = model(frame, verbose=False, device="cpu")
    return results[0].plot()


# ──────────────────────────────────────────────────────────────────────────────
# Single-thread processing
# ──────────────────────────────────────────────────────────────────────────────

def process_single(video_path: str, output_path: str) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.fps, reader.width, reader.height)
    model = YOLO("yolov8s-pose.pt")
    model.to("cpu")

    t_start = time.perf_counter()

    idx = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        annotated = run_inference(model, frame)
        writer.write(annotated)
        idx += 1

    elapsed = time.perf_counter() - t_start
    return elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Multi-thread processing
# ──────────────────────────────────────────────────────────────────────────────

_POISON = None   # sentinel value to stop workers


def worker_fn(in_q: queue.Queue, out_q: queue.Queue):
    """Worker thread: each thread owns its own YOLO instance (thread-safe)."""
    model = YOLO("yolov8s-pose.pt")
    model.to("cpu")
    while True:
        item = in_q.get()
        if item is _POISON:
            out_q.put(_POISON)   # propagate sentinel
            in_q.task_done()
            break
        idx, frame = item
        annotated = run_inference(model, frame)
        out_q.put((idx, annotated))
        in_q.task_done()


def process_multi(video_path: str, output_path: str, num_workers: int) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.fps, reader.width, reader.height)

    in_q:  queue.Queue = queue.Queue(maxsize=num_workers * 4)
    out_q: queue.Queue = queue.Queue()

    # Start workers
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker_fn, args=(in_q, out_q), daemon=True)
        t.start()
        threads.append(t)

    t_start = time.perf_counter()

    # Producer thread: read all frames into in_q
    def producer():
        frame_idx = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            in_q.put((frame_idx, frame))
            frame_idx += 1
        # Send poison pills
        for _ in range(num_workers):
            in_q.put(_POISON)

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    # Consumer: collect results and restore order
    pending: dict[int, np.ndarray] = {}
    next_to_write = 0
    finished_workers = 0

    while finished_workers < num_workers:
        item = out_q.get()
        if item is _POISON:
            finished_workers += 1
            continue
        idx, frame = item
        pending[idx] = frame

        # Write all consecutive frames we have
        while next_to_write in pending:
            writer.write(pending.pop(next_to_write))
            next_to_write += 1

    # Flush any remaining frames (should be empty after proper sync)
    while next_to_write in pending:
        writer.write(pending.pop(next_to_write))
        next_to_write += 1

    prod.join()
    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t_start
    return elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8s-pose video inference (single or multi thread)"
    )
    parser.add_argument("--video_path", required=True,
                        help="Path to input video (640x480)")
    parser.add_argument("--mode", choices=["single", "multi"],
                        default="single",
                        help="Execution mode: single or multi thread")
    parser.add_argument("--output", required=True,
                        help="Output video filename")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads (only in multi mode)")
    args = parser.parse_args()

    print(f"Mode      : {args.mode}")
    if args.mode == "multi":
        print(f"Workers   : {args.workers}")

    if args.mode == "single":
        elapsed = process_single(args.video_path, args.output)
    else:
        elapsed = process_multi(args.video_path, args.output, args.workers)

    print(f"Time      : {elapsed:.3f} s")
    print(f"Output    : {args.output}")


if __name__ == "__main__":
    main()
