from queue import Queue, Empty
import logging
import argparse
import threading
import time
import cv2
import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"


# Настройка логирования
os.makedirs("log", exist_ok=True)

logging.basicConfig(
    filename="log/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class Sensor:
    def get(self):
        raise NotImplementedError()


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam:
    def __init__(self, cam_name: int, resolution: tuple):
        self.cam_name = cam_name
        self.width, self.height = resolution
        self.cap = None

        try:
            self.cap = cv2.VideoCapture(cam_name)

            if not self.cap.isOpened():
                raise RuntimeError(f"Camera {cam_name} not found")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            logging.info(f"Camera initialized: {cam_name}")

        except Exception as e:
            logging.error(f"Camera init error: {e}")
            raise

    def get(self):
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret:
            logging.error("Camera read error")
            return None
        return frame

    def __del__(self):
        try:
            if self.cap:
                self.cap.release()
                logging.info("Camera released")
        except Exception as e:
            logging.error(f"Camera release error: {e}")


class WindowImage:
    def __init__(self, freq_hz: float):
        self.delay = 1.0 / freq_hz
        self.window_name = "Sensor System"

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        logging.info("Window initialized")

    def show(self, img):
        try:
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)
            time.sleep(self.delay)
        except Exception as e:
            logging.error(f"Window show error: {e}")

    def __del__(self):
        try:
            cv2.destroyAllWindows()
            logging.info("Window destroyed")
        except Exception as e:
            logging.error(f"Window destroy error: {e}")


def sensor_worker(sensor, queue: Queue, stop_event: threading.Event):
    """Запускается в отдельном потоке, бесконечно получает данные от датчика и кладет их в очередь"""
    try:
        while not stop_event.is_set():
            data = sensor.get()

            if queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    pass

            queue.put(data)

    except Exception as e:
        logging.error(f"Sensor worker error: {e}")
        stop_event.set()


def camera_worker(cam: SensorCam, queue: Queue, stop_event: threading.Event):
    """Запускается в отдельном потоке, захватывает кадры с камеры и кладет их в очередь"""
    try:
        while not stop_event.is_set():
            frame = cam.get()

            if frame is None:
                stop_event.set()
                print("Camera read error")
                break

            if queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    pass

            queue.put(frame)

    except Exception as e:
        logging.error(f"Camera worker error: {e}")
        print("There was a problem with the camera")
        stop_event.set()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cam", type=str, default="0", help="Camera name")
    parser.add_argument("--res", type=str,
                        default="640x480", help="Resolution WxH")
    parser.add_argument("--fps", type=float, default=30,
                        help="Display frequency")

    args = parser.parse_args()

    w, h = map(int, args.res.split("x"))

    stop_event = threading.Event()

    cam_queue = Queue(maxsize=1)
    s1_queue = Queue(maxsize=1)
    s2_queue = Queue(maxsize=1)
    s3_queue = Queue(maxsize=1)

    try:
        cam = SensorCam(int(args.cam), (w, h))
    except RuntimeError:
        print("Camera not found")
        return

    s1 = SensorX(0.01)
    s2 = SensorX(0.1)
    s3 = SensorX(1.0)

    window = WindowImage(args.fps)

    threads = [
        threading.Thread(target=camera_worker, args=(
            cam, cam_queue, stop_event)),
        threading.Thread(target=sensor_worker,
                         args=(s1, s1_queue, stop_event)),
        threading.Thread(target=sensor_worker,
                         args=(s2, s2_queue, stop_event)),
        threading.Thread(target=sensor_worker,
                         args=(s3, s3_queue, stop_event)),
    ]

    for t in threads:
        t.start()

    try:
        last_s1 = last_s2 = last_s3 = 0

        while not stop_event.is_set():
            try:
                frame = cam_queue.get(timeout=1)
            except Empty:
                continue

            try:
                last_s1 = s1_queue.get_nowait()
            except Empty:
                pass

            try:
                last_s2 = s2_queue.get_nowait()
            except Empty:
                pass

            try:
                last_s3 = s3_queue.get_nowait()
            except Empty:
                pass

            cv2.putText(frame, f"S1 (100Hz): {last_s1}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"S2 (10Hz): {last_s2}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, f"S3 (1Hz): {last_s3}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            window.show(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        stop_event.set()

        for t in threads:
            t.join(timeout=2)

        del cam
        del window

        logging.info("Program stopped safely")


if __name__ == "__main__":
    main()
