from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
from multiprocessing import Event, Queue, Value

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8s-pose.pt"


class Camera:
    """Захват кадров с веб-камеры."""

    def __init__(self, camera_name: str, width: int, height: int) -> None:
        """Открыть камеру и задать размер кадра."""
        camera_id = int(camera_name) if camera_name.isdigit() else camera_name
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_name}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        """Считать кадр; (ok, frame)."""
        return self._cap.read()

    def release(self) -> None:
        """Освободить камеру."""
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()
        self._cap = None


class Window:
    """Окно отображения результата."""

    def __init__(self, name: str = "YOLOv8 pose realtime") -> None:
        """Создать именованное окно OpenCV."""
        self._name = name
        self._closed = False
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def show(self, frame) -> bool:
        """Показать кадр; False при нажатии q."""
        cv2.imshow(self._name, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        """Закрыть окно."""
        if self._closed:
            return
        try:
            cv2.destroyWindow(self._name)
        except cv2.error:
            pass
        self._closed = True


def put_latest(out_queue: Queue, value) -> None:
    """Положить значение в очередь, отбрасывая старые при переполнении."""
    for _ in range(32):
        try:
            out_queue.put_nowait(value)
            return
        except queue.Full:
            try:
                out_queue.get_nowait()
            except queue.Empty:
                return


def default_workers(requested: int | None) -> int:
    """Число процессов-воркеров для CPU."""
    if requested is not None:
        return max(1, requested)
    return min(4, mp.cpu_count() or 4)


def _predict_kwargs(imgsz: int) -> dict:
    """Собрать kwargs для вызова model() на CPU."""
    return {"verbose": False, "device": "cpu", "imgsz": imgsz}


def signal_workers_stop(input_queue: Queue, count: int) -> None:
    """Отправить каждому воркеру сигнал завершения (None)."""
    for _ in range(count):
        while True:
            try:
                input_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    input_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)


def worker_process(
    worker_id: int,
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    imgsz: int,
    infer_fps: Value,
    worker_ready: Event,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    model = YOLO(MODEL_NAME)
    predict_kw = _predict_kwargs(imgsz)
    print(f"Worker {worker_id}: {MODEL_NAME} on cpu, imgsz={imgsz}")

    import numpy as np

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, **predict_kw)
    worker_ready.set()

    infer_frames = 0
    infer_t0 = time.perf_counter()

    while not stop_event.is_set():
        try:
            frame_data = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if frame_data is None:
            break

        frame_id, frame = frame_data
        try:
            result = model(frame, **predict_kw)[0]
            put_latest(output_queue, (frame_id, result.plot()))
        except Exception as exc:
            print(f"Worker {worker_id} inference error: {exc}")
            continue

        infer_frames += 1
        now = time.perf_counter()
        if now - infer_t0 >= 1.0:
            with infer_fps.get_lock():
                infer_fps.value = int(infer_frames / (now - infer_t0))
            infer_frames = 0
            infer_t0 = now

    print(f"Worker {worker_id} stopped")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    cpu_default = min(4, mp.cpu_count() or 4)
    parser = argparse.ArgumentParser(
        description="Realtime yolov8-pose (CPU multiprocessing)"
    )
    parser.add_argument("--camera", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Worker processes (default: {cpu_default})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="YOLO inference size",
    )
    parser.add_argument("--queue-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    """Запустить камеру, воркеры и цикл отображения pose."""
    args = parse_args()
    workers_count = default_workers(args.workers)

    print(
        f"Model: {MODEL_NAME}, device=cpu, "
        f"workers={workers_count}, imgsz={args.imgsz}"
    )

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    camera = Camera(args.camera, args.width, args.height)
    window = Window()
    stop_event = Event()
    infer_fps = Value("i", 0)

    input_queue: Queue = Queue(maxsize=args.queue_size)
    output_queue: Queue = Queue(maxsize=2)

    ready_events = [Event() for _ in range(workers_count)]
    workers: list[mp.Process] = []

    for i in range(workers_count):
        proc = mp.Process(
            target=worker_process,
            args=(
                i,
                input_queue,
                output_queue,
                stop_event,
                args.imgsz,
                infer_fps,
                ready_events[i],
            ),
        )
        workers.append(proc)
        proc.start()

    print("Loading model in worker(s)...")
    for i, ready in enumerate(ready_events):
        if not ready.wait(timeout=180):
            raise RuntimeError(f"Worker {i} did not become ready in time")

    frame_id = 0
    latest_by_id: dict[int, object] = {}
    display_frames = 0
    last_fps_update = time.perf_counter()
    current_disp_fps = 0.0
    current_infer_fps = 0

    print("Ready. Press q in the window to quit.")

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                print("Camera read error")
                break

            frame_id += 1
            put_latest(input_queue, (frame_id, frame))

            while True:
                try:
                    rid, result_frame = output_queue.get_nowait()
                except queue.Empty:
                    break
                latest_by_id[rid] = result_frame
                if len(latest_by_id) > 8:
                    del latest_by_id[min(latest_by_id)]

            shown = latest_by_id[max(latest_by_id)] if latest_by_id else frame

            display_frames += 1
            now = time.perf_counter()
            if now - last_fps_update >= 1.0:
                current_disp_fps = display_frames / (now - last_fps_update)
                display_frames = 0
                last_fps_update = now
                with infer_fps.get_lock():
                    current_infer_fps = infer_fps.value

            overlay = shown.copy()
            cv2.putText(
                overlay,
                f"Display FPS: {current_disp_fps:.1f} | Inference FPS: {current_infer_fps}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if not window.show(overlay):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Shutting down...")
        stop_event.set()
        signal_workers_stop(input_queue, workers_count)

        for proc in workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

        window.close()
        camera.release()
        print("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
