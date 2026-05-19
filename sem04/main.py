"""
Задание 1: Формирование управляющего сигнала
Запуск: python main.py --camera /dev/video0 --resolution 1280x720 --fps 30 --mode thread
         python main.py --camera /dev/video0 --resolution 1280x720 --fps 30 --mode process
"""
import argparse
import logging
import os
import time
import threading
import multiprocessing
import queue
from abc import ABC, abstractmethod

import cv2
import numpy as np

# ───────────────────────── логирование ──────────────────────────
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("log/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")


# ───────────────────────── базовый датчик ───────────────────────
class Sensor(ABC):
    @abstractmethod
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


# ───────────────────────── SensorX (НЕ МЕНЯТЬ!) ─────────────────
class SensorX(Sensor):
    """Sensor X"""
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


# ───────────────────────── SensorCam (RAII) ─────────────────────
class SensorCam(Sensor):
    """USB-камера. RAII: инициализация в __init__, освобождение в __del__."""

    def __init__(self, camera_id: str, resolution: str):
        self._logger = logging.getLogger("SensorCam")
        self._camera_id = camera_id
        self._resolution = resolution

        # разбираем разрешение
        try:
            w, h = map(int, resolution.split("x"))
        except ValueError:
            self._logger.error("Неверный формат разрешения: %s", resolution)
            raise

        # числовой индекс или путь к устройству
        try:
            cam_index = int(camera_id)
        except ValueError:
            cam_index = camera_id

        self._cap = cv2.VideoCapture(cam_index)

        if not self._cap.isOpened():
            self._logger.error("Камера %s не найдена в системе", camera_id)
            raise RuntimeError(f"Камера {camera_id} не найдена")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._logger.info("Камера %s открыта, разрешение %dx%d", camera_id, w, h)

    def get(self):
        """Возвращает очередной кадр или None при ошибке."""
        ret, frame = self._cap.read()
        if not ret:
            self._logger.error("Ошибка чтения кадра (камера %s)", self._camera_id)
            return None
        return frame

    def __del__(self):
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()
            logging.getLogger("SensorCam").info(
                "Камера %s освобождена", self._camera_id
            )


# ───────────────────────── WindowImage (RAII) ───────────────────
class WindowImage:
    """Окно отображения. RAII: создание в __init__, уничтожение в __del__."""

    WIN_NAME = "Sensor Dashboard"

    def __init__(self, fps: float):
        self._logger = logging.getLogger("WindowImage")
        self._fps = fps
        self._delay_ms = max(1, int(1000 / fps))

        try:
            cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
            self._logger.info("Окно создано, FPS=%s", fps)
        except Exception as exc:
            self._logger.error("Ошибка создания окна: %s", exc)
            raise

    def show(self, img: np.ndarray) -> int:
        """Показать img. Возвращает код нажатой клавиши."""
        cv2.imshow(self.WIN_NAME, img)
        return cv2.waitKey(self._delay_ms)

    def __del__(self):
        try:
            cv2.destroyWindow(self.WIN_NAME)
            logging.getLogger("WindowImage").info("Окно уничтожено")
        except Exception:
            pass


# ─────────────── воркеры для потоков ────────────────────────────
def sensor_thread_worker(sensor: Sensor, q: queue.Queue, stop_event: threading.Event):
    """Бесконечно опрашивает датчик и кладёт значение в очередь размера 1."""
    while not stop_event.is_set():
        value = sensor.get()
        # очередь размером 1 — всегда самое свежее значение
        if not q.full():
            q.put_nowait(value)
        else:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(value)
            except queue.Full:
                pass


def cam_thread_worker(sensor: SensorCam, q: queue.Queue, stop_event: threading.Event):
    """Воркер камеры: кладёт кадры в очередь размера 1."""
    while not stop_event.is_set():
        frame = sensor.get()
        if frame is None:
            continue
        if not q.full():
            q.put_nowait(frame)
        else:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass


# ─────────────── воркеры для процессов ──────────────────────────
def sensor_process_worker(delay: float, mp_q: multiprocessing.Queue, stop_event):
    """Процессный воркер для SensorX."""
    sensor = SensorX(delay)
    while not stop_event.is_set():
        value = sensor.get()
        # обнуляем очередь, кладём только последнее
        while not mp_q.empty():
            try:
                mp_q.get_nowait()
            except Exception:
                break
        mp_q.put(value)


def cam_process_worker(camera_id: str, resolution: str, mp_q: multiprocessing.Queue, stop_event):
    """Процессный воркер для камеры."""
    log = logging.getLogger("cam_process")
    try:
        cam = SensorCam(camera_id, resolution)
    except Exception as exc:
        log.error("Процесс камеры: ошибка инициализации: %s", exc)
        return
    while not stop_event.is_set():
        frame = cam.get()
        if frame is None:
            continue
        while not mp_q.empty():
            try:
                mp_q.get_nowait()
            except Exception:
                break
        mp_q.put(frame)


# ─────────────── наложение данных датчиков на кадр ──────────────
def overlay_sensors(frame: np.ndarray, sx_values: list) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 200, h - 90), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    labels = ["Sensor0", "Sensor1", "Sensor2"]
    for i, val in enumerate(sx_values):
        cv2.putText(
            img,
            f"{labels[i]}: {val}",
            (w - 190, h - 70 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


# ─────────────── режим ПОТОКОВ ──────────────────────────────────
def run_threads(camera_id: str, resolution: str, fps: float):
    logger.info("=== Режим: ПОТОКИ ===")
    stop_event = threading.Event()

    cam = SensorCam(camera_id, resolution)
    sensors = [SensorX(0.01), SensorX(0.1), SensorX(1.0)]

    cam_q = queue.Queue(maxsize=1)
    sx_qs = [queue.Queue(maxsize=1) for _ in sensors]

    # запуск потоков
    threads = []
    t = threading.Thread(target=cam_thread_worker, args=(cam, cam_q, stop_event), daemon=True)
    threads.append(t)
    for i, s in enumerate(sensors):
        t = threading.Thread(target=sensor_thread_worker, args=(s, sx_qs[i], stop_event), daemon=True)
        threads.append(t)
    for t in threads:
        t.start()

    window = WindowImage(fps)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    last_frame = blank
    last_sx = [0, 0, 0]

    try:
        while True:
            # получаем кадр
            try:
                last_frame = cam_q.get_nowait()
            except queue.Empty:
                pass

            # получаем значения датчиков
            for i, q_ in enumerate(sx_qs):
                try:
                    last_sx[i] = q_.get_nowait()
                except queue.Empty:
                    pass

            img = overlay_sensors(last_frame, last_sx)
            key = window.show(img)
            if key == ord("q"):
                logger.info("Нажата 'q', завершение...")
                break
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        logger.info("Все потоки остановлены")


# ─────────────── режим ПРОЦЕССОВ ────────────────────────────────
def run_processes(camera_id: str, resolution: str, fps: float):
    logger.info("=== Режим: ПРОЦЕССЫ ===")
    stop_event = multiprocessing.Event()

    cam_q = multiprocessing.Queue(maxsize=2)
    sx_qs = [multiprocessing.Queue(maxsize=2) for _ in range(3)]
    delays = [0.01, 0.1, 1.0]

    processes = []
    p = multiprocessing.Process(
        target=cam_process_worker, args=(camera_id, resolution, cam_q, stop_event), daemon=True
    )
    processes.append(p)
    for i, d in enumerate(delays):
        p = multiprocessing.Process(
            target=sensor_process_worker, args=(d, sx_qs[i], stop_event), daemon=True
        )
        processes.append(p)
    for p in processes:
        p.start()

    window = WindowImage(fps)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    last_frame = blank
    last_sx = [0, 0, 0]

    try:
        while True:
            try:
                last_frame = cam_q.get_nowait()
            except Exception:
                pass

            for i, q_ in enumerate(sx_qs):
                try:
                    last_sx[i] = q_.get_nowait()
                except Exception:
                    pass

            img = overlay_sensors(last_frame, last_sx)
            key = window.show(img)
            if key == ord("q"):
                logger.info("Нажата 'q', завершение...")
                break
    finally:
        stop_event.set()
        for p in processes:
            p.join(timeout=2)
        logger.info("Все процессы остановлены")


# ─────────────── argparse + точка входа ─────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Sensor Dashboard")
    parser.add_argument(
        "--camera", default="0",
        help="Имя/индекс камеры (напр. /dev/video0 или 0)"
    )
    parser.add_argument(
        "--resolution", default="640x480",
        help="Разрешение камеры, напр. 1280x720"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Частота отображения картинки (FPS)"
    )
    parser.add_argument(
        "--mode", choices=["thread", "process"], default="thread",
        help="thread — потоки (по заданию), process — процессы (эксперимент)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(
        "Запуск: camera=%s, resolution=%s, fps=%s, mode=%s",
        args.camera, args.resolution, args.fps, args.mode,
    )
    if args.mode == "thread":
        run_threads(args.camera, args.resolution, args.fps)
    else:
        run_processes(args.camera, args.resolution, args.fps)
