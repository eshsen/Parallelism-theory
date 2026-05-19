import cv2

# Проверить все доступные камеры
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Камера {i} доступна")
        cap.release()
    else:
        print(f"Камера {i} не доступна")
