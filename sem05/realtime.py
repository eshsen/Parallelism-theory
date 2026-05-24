"""
Real-time YOLOv8s-pose inference from webcam with multi-thread acceleration.

Usage:
    python realtime.py
    python realtime.py --workers 4 --camera 0
    python realtime.py --workers 4 --camera 0 --output cam_output.mp4

Press 'q' to quit.
"""

import argparse
import time
import queue
import threading
import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# RAII wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CameraCapture:
    """RAII wrapper around cv2.VideoCapture for a camera."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    def write(self, frame: np.ndarray):
        if self._writer.isOpened():
            self._writer.write(frame)

    def __del__(self):
        if self._writer.isOpened():
            self._writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────────────

def worker_fn(in_q: queue.Queue, out_q: queue.Queue):
    """Worker: owns its own YOLO model instance (thread-safe per Ultralytics docs)."""
    model = YOLO("yolov8s-pose.pt")
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        idx, frame = item
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        out_q.put((idx, annotated))


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8s-pose from webcam"
    )
    parser.add_argument("--camera",  type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of worker threads (default: 2)")
    parser.add_argument("--output",  type=str, default=None,
                        help="Optional: save output to this video file")
    args = parser.parse_args()

    cam    = CameraCapture(args.camera)
    writer = VideoWriter(args.output, cam.fps, cam.width, cam.height) \
             if args.output else None

    # Queues: small buffers to reduce latency
    in_q:  queue.Queue = queue.Queue(maxsize=args.workers * 2)
    out_q: queue.Queue = queue.Queue(maxsize=args.workers * 2)

    # Start workers
    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker_fn, args=(in_q, out_q), daemon=True)
        t.start()
        threads.append(t)

    # FPS counter
    fps_display = 0.0
    fps_counter = 0
    fps_t0      = time.perf_counter()

    # Latest processed frame (for display when output is behind input)
    latest_frame: np.ndarray | None = None
    frame_idx   = 0
    stop_event  = threading.Event()

    def producer():
        nonlocal frame_idx
        while not stop_event.is_set():
            ok, frame = cam.read()
            if not ok:
                break
            # Drop frame if queue is full (keep latency low)
            try:
                in_q.put_nowait((frame_idx, frame))
                frame_idx += 1
            except queue.Full:
                pass
        # Send poison pills
        for _ in range(args.workers):
            in_q.put(None)

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    print(f"Running with {args.workers} worker thread(s). Press 'q' to quit.")

    finished_workers = 0
    while finished_workers < args.workers:
        try:
            item = out_q.get(timeout=0.1)
        except queue.Empty:
            # Show last known frame while waiting
            if latest_frame is not None:
                cv2.imshow("Pose (real-time)", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
            continue

        if item is None:
            finished_workers += 1
            continue

        _, annotated = item
        latest_frame = annotated

        # FPS calculation
        fps_counter += 1
        now = time.perf_counter()
        if now - fps_t0 >= 1.0:
            fps_display = fps_counter / (now - fps_t0)
            fps_counter = 0
            fps_t0      = now

        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Pose (real-time)", annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    prod.join(timeout=2)
    for t in threads:
        t.join(timeout=2)

    print("Done.")


if __name__ == "__main__":
    main()
