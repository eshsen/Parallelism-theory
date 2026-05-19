"""
Бенчмарк: сравнение задержек потоков vs процессов для SensorX.
Камера в бенчмарке не используется (нет гарантии наличия устройства).
Результаты сохраняются в benchmark_results.csv и строятся в graphs.ipynb.

Запуск: python benchmark.py
"""
import time
import threading
import multiprocessing
import queue
import statistics
import csv
import os

# SensorX — не модифицируется
class SensorX:
    """Sensor X"""
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


DELAYS = [0.01, 0.1, 1.0]   # 100 Hz, 10 Hz, 1 Hz
N_SAMPLES = 30               # количество измерений на каждый датчик
WARMUP = 3                   # прогревочных итераций


# ── потоковый воркер ──────────────────────────────────────────────
def _thread_worker(delay, result_q: queue.Queue, stop: threading.Event, timestamps: list):
    sensor = SensorX(delay)
    while not stop.is_set():
        t_before = time.perf_counter()
        val = sensor.get()
        t_after = time.perf_counter()
        timestamps.append((t_before, t_after, val))
        # кладём в очередь только последнее
        if not result_q.full():
            result_q.put_nowait((val, time.perf_counter()))
        else:
            try:
                result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                result_q.put_nowait((val, time.perf_counter()))
            except queue.Full:
                pass


# ── процессный воркер ─────────────────────────────────────────────
def _process_worker(delay, mp_q: multiprocessing.Queue, stop_event, mp_timestamps):
    sensor = SensorX(delay)
    while not stop_event.is_set():
        t_before = time.perf_counter()
        val = sensor.get()
        t_after = time.perf_counter()
        mp_timestamps.append((t_before, t_after, val))
        while not mp_q.empty():
            try:
                mp_q.get_nowait()
            except Exception:
                break
        mp_q.put((val, time.perf_counter()))


def measure_thread(delay: float, n_samples: int):
    """Запускает воркер-поток и измеряет latency доставки в главный поток."""
    result_q = queue.Queue(maxsize=1)
    stop = threading.Event()
    timestamps = []

    t = threading.Thread(
        target=_thread_worker,
        args=(delay, result_q, stop, timestamps),
        daemon=True,
    )
    t.start()

    latencies = []
    collected = 0
    # прогрев
    for _ in range(WARMUP):
        try:
            result_q.get(timeout=delay * 5)
        except queue.Empty:
            pass

    while collected < n_samples:
        try:
            val, t_put = result_q.get(timeout=delay * 5)
            t_get = time.perf_counter()
            latencies.append((t_get - t_put) * 1000)  # мс
            collected += 1
        except queue.Empty:
            pass

    stop.set()
    t.join(timeout=2)
    return latencies


def measure_process(delay: float, n_samples: int):
    """Запускает процесс-воркер и измеряет latency."""
    mp_q = multiprocessing.Queue(maxsize=2)
    stop_event = multiprocessing.Event()
    mp_timestamps = multiprocessing.Manager().list()

    p = multiprocessing.Process(
        target=_process_worker,
        args=(delay, mp_q, stop_event, mp_timestamps),
        daemon=True,
    )
    p.start()

    latencies = []
    collected = 0
    # прогрев
    for _ in range(WARMUP):
        try:
            mp_q.get(timeout=delay * 5)
        except Exception:
            pass

    while collected < n_samples:
        try:
            val, t_put = mp_q.get(timeout=delay * 5)
            t_get = time.perf_counter()
            latencies.append((t_get - t_put) * 1000)  # мс
            collected += 1
        except Exception:
            pass

    stop_event.set()
    p.join(timeout=3)
    return latencies


def run_benchmark():
    rows = []
    print(f"{'Delay':>8} | {'Mode':>8} | {'Mean ms':>10} | {'Std ms':>10} | {'Min ms':>10} | {'Max ms':>10}")
    print("-" * 65)

    for delay in DELAYS:
        freq = round(1 / delay)
        for mode, measure_fn in [("thread", measure_thread), ("process", measure_process)]:
            lats = measure_fn(delay, N_SAMPLES)
            if not lats:
                print(f"{delay:>8} | {mode:>8} | NO DATA")
                continue
            mean_ = statistics.mean(lats)
            std_  = statistics.stdev(lats) if len(lats) > 1 else 0.0
            min_  = min(lats)
            max_  = max(lats)
            print(f"{delay:>8} | {mode:>8} | {mean_:>10.4f} | {std_:>10.4f} | {min_:>10.4f} | {max_:>10.4f}")
            rows.append({
                "delay": delay,
                "freq_hz": freq,
                "mode": mode,
                "mean_ms": round(mean_, 5),
                "std_ms":  round(std_,  5),
                "min_ms":  round(min_,  5),
                "max_ms":  round(max_,  5),
                "n_samples": len(lats),
            })

    # сохранить CSV
    csv_path = "benchmark_results.csv"
    fieldnames = ["delay", "freq_hz", "mode", "mean_ms", "std_ms", "min_ms", "max_ms", "n_samples"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nРезультаты сохранены в {csv_path}")
    return rows


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_benchmark()
