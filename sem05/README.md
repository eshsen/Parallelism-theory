# YOLOv8s-pose — параллельный инференс на CPU

---

## Использование

### 1. Однопоточная обработка видео

```bash
python main.py --video_path input.mp4 --mode single --output out_single.mp4
```

### 2. Многопоточная обработка (4 потока)

```bash
python main.py --video_path input.mp4 --mode multi --output out_multi.mp4 --workers 4
```

### 3. Подбор оптимального числа потоков (бенчмарк)

```bash
python benchmark.py --video_path input.mp4 --max_workers 8
```

### 4. Real-time с камеры

```bash
python realtime.py --workers 4 --camera 0
# С записью результата:
python realtime.py --workers 4 --camera 0 --output cam_result.mp4
```

Нажмите `q` для выхода.

---
