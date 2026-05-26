import cv2
import time
import threading
from queue import Queue
from ultralytics import YOLO


class CameraRAII:
    """RAII обертка для камеры. Чтение кадров вынесено в фоновый поток."""

    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        self.running = False
        self.frame_queue = Queue(maxsize=1)

    def start(self):
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()  # захват кадра с камеры
            if not ret:
                break
            if not self.frame_queue.empty():  # если в очереди есть старый кадр
                self.frame_queue.get_nowait()  # удаляем его
            self.frame_queue.put_nowait(frame)  # кладем свежий в очередь

    def get_frame(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def stop(self):
        self.running = False
        self.cap.release()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run_realtime():
    model = YOLO('yolov8s-pose.pt')
    fps_history = []

    with CameraRAII(0) as cam:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            t0 = time.time()
            results = model(frame, verbose=False)
            out_frame = results[0].plot()
            fps = 1.0 / (time.time() - t0)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            cv2.putText(out_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Realtime Pose", out_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_realtime()
