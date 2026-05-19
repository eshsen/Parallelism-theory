# Задание 1 — Python Thread/Process: Формирование управляющего сигнала

## Файлы проекта

| Файл | Назначение |
|---|---|
| `main.py` | Основная программа: датчики, камера, окно, RAII, argparse |
| `benchmark.py` | Измерение задержек потоки vs процессы, генерирует CSV |
| `graphs.ipynb` | Jupyter-ноутбук с графиками ускорения и эффективности |
| `requirements.txt` | Зависимости |
| `log/` | Лог-файлы (создаётся автоматически) |

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск основной программы

```bash
# Режим потоков (по заданию):
python main.py --camera 0 --resolution 640x480 --fps 30 --mode thread

# Режим процессов (эксперимент для сравнения задержек):
python main.py --camera /dev/video0 --resolution 1280x720 --fps 25 --mode process
```

**Клавиша `q`** — безопасное завершение программы.

## Запуск бенчмарка (без камеры)

```bash
python benchmark.py
# → создаёт benchmark_results.csv
```

## Просмотр графиков

```bash
jupyter notebook graphs.ipynb
```
Или открыть `graphs.ipynb` в VS Code / JupyterLab.

## Архитектура

```
main.py
├── Sensor (ABC)
│   ├── SensorX(delay)          — не модифицируется!
│   └── SensorCam(cam_id, res)  — RAII, OpenCV, logging
├── WindowImage(fps)             — RAII, show(img), logging
├── run_threads()                — threading + queue.Queue(maxsize=1)
└── run_processes()              — multiprocessing + Queue(maxsize=2)
```

### Принцип "только последнее значение"

Очереди создаются с `maxsize=1`.  
Воркер при заполненной очереди удаляет старое и кладёт новое — главный поток всегда получает самые свежие данные (минимальная задержка).

## Выводы по задержкам

| Датчик | Потоки (мс) | Процессы (мс) | Ускорение |
|---|---|---|---|
| SensorX 100 Hz | ~0.10 | ~0.18 | ~1.8× |
| SensorX 10 Hz  | ~0.14 | ~0.19 | ~1.4× |
| SensorX 1 Hz   | ~0.13 | ~0.21 | ~1.6× |

Для I/O-bound датчиков **потоки быстрее процессов** из-за отсутствия IPC-накладных расходов.
