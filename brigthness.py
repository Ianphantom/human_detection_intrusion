import cv2
import time
import numpy as np
import csv
import torch
from torchvision import transforms

# Load model YOLOv5 (pastikan sudah install torch dan model YOLOv5 sudah ada)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# Buat file CSV untuk log hasil uji
csv_file = open('uji_pencahayaan_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'brightness', 'detected_person_count', 'fps'])

cap = cv2.VideoCapture(0)  # Webcam

prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        brightness = get_brightness(frame)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(img_rgb)

        # Count detected persons (class 0 biasanya person)
        persons = [x for x in results.xyxy[0] if int(x[5]) == 0]
        person_count = len(persons)

        # Hitung FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-5)

        # Tentukan kondisi pencahayaan berdasar brightness
        if brightness < 50:
            lighting_condition = "Gelap"
        elif brightness < 150:
            lighting_condition = "Sedang"
        else:
            lighting_condition = "Terang"

        # Gambar bounding box orang
        for *box, conf, cls in persons:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay info ke frame
        cv2.putText(frame, f'Pencahayaan: {lighting_condition} ({brightness:.1f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Jumlah Orang: {person_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Tampilkan frame
        cv2.imshow('Uji Pencahayaan YOLOv5', frame)

        # Simpan log ke CSV
        csv_writer.writerow([time.time(), brightness, person_count, fps])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
