"""
Benchmark: measure processing time for 1..max_workers threads on a video file.
Saves results to benchmark_results.csv for use in graphs.ipynb.

Usage:
    python benchmark.py --video_path input.mp4 --max_workers 8
"""

import argparse
import csv
import os
import time
import queue
import threading
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# RAII wrappers (same as in main.py)
# ──────────────────────────────────────────────────────────────────────────────

class VideoReader:
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
    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def __del__(self):
        if self._writer.isOpened():
            self._writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Processing functions (same logic as main.py)
# ──────────────────────────────────────────────────────────────────────────────

def process_single(video_path: str, out_path: str) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(out_path, reader.fps, reader.width, reader.height)
    model = YOLO("yolov8s-pose.pt")
    t0 = time.perf_counter()
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        results = model(frame, verbose=False, device="cpu")
        writer.write(results[0].plot())
    return time.perf_counter() - t0


def _worker(in_q, out_q):
    model = YOLO("yolov8s-pose.pt")
    model.to("cpu")
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        idx, frame = item
        results = model(frame, verbose=False, device="cpu")
        out_q.put((idx, results[0].plot()))


def process_multi(video_path: str, out_path: str, num_workers: int) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(out_path, reader.fps, reader.width, reader.height)
    in_q:  queue.Queue = queue.Queue(maxsize=num_workers * 4)
    out_q: queue.Queue = queue.Queue()

    threads = [
        threading.Thread(target=_worker, args=(in_q, out_q), daemon=True)
        for _ in range(num_workers)
    ]
    for t in threads:
        t.start()

    t0 = time.perf_counter()

    def producer():
        fi = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            in_q.put((fi, frame))
            fi += 1
        for _ in range(num_workers):
            in_q.put(None)

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    pending: dict = {}
    next_write = 0
    done = 0
    while done < num_workers:
        item = out_q.get()
        if item is None:
            done += 1
            continue
        idx, frame = item
        pending[idx] = frame
        while next_write in pending:
            writer.write(pending.pop(next_write))
            next_write += 1

    elapsed = time.perf_counter() - t0
    prod.join()
    for t in threads:
        t.join()
    return elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8s-pose throughput vs thread count"
    )
    parser.add_argument("--video_path",  required=True)
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Max number of worker threads to test")
    parser.add_argument("--output_csv",  default="benchmark_results.csv",
                        help="Path to save CSV results")
    args = parser.parse_args()

    results = []

    # Single thread (baseline)
    print("workers=1 (single thread) ...", end=" ", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp = f.name
    t1 = process_single(args.video_path, tmp)
    os.unlink(tmp)
    print(f"{t1:.2f} s")
    results.append({"workers": 1, "time_s": t1,
                    "speedup": 1.0, "efficiency": 1.0})

    # Multi-thread sweep
    for n in range(2, args.max_workers + 1):
        print(f"workers={n} ...", end=" ", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        tn = process_multi(args.video_path, tmp, n)
        os.unlink(tmp)
        speedup = t1 / tn
        efficiency = speedup / n
        print(f"{tn:.2f} s  speedup={speedup:.2f}  eff={efficiency:.2f}")
        results.append({"workers": n, "time_s": tn,
                        "speedup": speedup, "efficiency": efficiency})

    # Save CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["workers", "time_s", "speedup", "efficiency"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output_csv}")
    best = min(results, key=lambda r: r["time_s"])
    print(f"Best config : {best['workers']} workers  "
          f"({best['time_s']:.2f} s, speedup={best['speedup']:.2f}x)")


if __name__ == "__main__":
    main()
