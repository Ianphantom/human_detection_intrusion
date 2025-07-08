import torch
import cv2
import os
import warnings
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image
import time
import requests
import json
from torchvision import transforms

# === Telegram Settings ===
TELEGRAM_BOT_TOKEN = '7585743264:AAGpvIaRIlJpEOLIfehCxew7F2ievdSgvZc'
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
ALERT_INTERVAL = 20 * 60  # 20 minutes
REMINDER_INTERVAL = 60    # seconds

telegram_chat_id = '1387459458'
last_alert_time = 0
alert_acknowledged = False
notification_feature_on = True
detection_pause_until = 0  # Timestamp until which detection is skipped

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.cpu()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# === Detection toggle ===
detection_on = False

def save_chat_id():
    global telegram_chat_id
    telegram_chat_id = chat_id_entry.get().strip()
    print(f"✅ Telegram Chat ID set to: {telegram_chat_id}")

def toggle_detection():
    global detection_on, last_alert_time, alert_acknowledged
    detection_on = not detection_on
    last_alert_time = 0  # Reset on toggle
    alert_acknowledged = False
    toggle_btn.config(text="Turn OFF Detection" if detection_on else "Turn ON Detection")

def send_telegram_alert(image_path):
    global last_alert_time, alert_acknowledged

    caption = "<b>🚨 Human Detected</b>\nClick below to acknowledge."
    with open(image_path, 'rb') as photo:
        data = {
            'chat_id': telegram_chat_id,
            'caption': caption,
            'parse_mode': 'HTML',
            'reply_markup': json.dumps({
                'inline_keyboard': [[
                    {'text': 'Acknowledge ✅', 'callback_data': 'ack_alert'}
                ]]
            })
        }
        files = {'photo': photo}
        response = requests.post(f'{TELEGRAM_API_URL}/sendPhoto', data=data, files=files)
        print(f'Telegram response: {response.status_code}')

def poll_telegram_updates():
    global alert_acknowledged, last_alert_time, detection_on
    offset = None
    while True:
        try:
            params = {'timeout': 10, 'offset': offset}
            response = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params=params)
            result = response.json().get("result", [])
            for update in result:
                offset = update["update_id"] + 1
                if 'callback_query' in update:
                    data = update['callback_query']['data']
                    if data == "ack_alert":
                        print("✅ Alert acknowledged.")
                        alert_acknowledged = True
                        detection_on = False
                        last_alert_time = time.time()
                        toggle_btn.config(text="Turn ON Detection")

                # Handle plain message
                elif 'message' in update:
                    message = update['message']
                    chat_id = str(message['chat']['id'])
                    if chat_id != telegram_chat_id:
                        continue  # Ignore if from unknown user

                    text = message.get('text', '').strip().upper()
                    if text == "TURN_ON":
                        detection_on = True
                        last_alert_time = 0
                        alert_acknowledged = False
                        print("🔔 Detection turned ON via Telegram")
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
                            'chat_id': telegram_chat_id,
                            'text': "✅ Detection is now ON."
                        })
                        toggle_btn.config(text="Turn OFF Detection" if detection_on else "Turn ON Detection")
                    elif text == "TURN_OFF":
                        detection_on = False
                        print("🛑 Detection turned OFF via Telegram")
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
                            'chat_id': telegram_chat_id,
                            'text': "❌ Detection is now OFF."
                        })
                        toggle_btn.config(text="Turn OFF Detection" if detection_on else "Turn ON Detection")
        except Exception as e:
            print("Polling error:", e)
        time.sleep(1)

def run_camera():
    global last_alert_time, alert_acknowledged, detection_on, detection_pause_until
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    prev_time = 0  # Initialize for FPS calculation

    while True:
        start_time = time.time()  # Frame start
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if detection_on and current_time >= detection_pause_until:
            results = model(frame)
            for *box, conf, cls in results.xyxy[0]:
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    current_time = time.time()
                    if notification_feature_on and not alert_acknowledged and (current_time - last_alert_time > REMINDER_INTERVAL):
                        img_path = 'human_detected.jpg'
                        cv2.imwrite(img_path, frame)
                        send_telegram_alert(img_path)
                        last_alert_time = current_time
                        detection_pause_until = current_time + 50
                        print("⏭️ Skipping detection for next 50 seconds.")
                        break
                    elif notification_feature_on and alert_acknowledged and (current_time - last_alert_time > ALERT_INTERVAL):
                        print("⏰ Reminder time passed, turning detection back ON.")
                        alert_acknowledged = False
                        detection_on = True
                        toggle_btn.config(text="Turn OFF Detection")
                    break

        # === FPS Calculation ===
        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-5)  # Add epsilon to avoid div by zero
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('YOLOv5 - Deteksi Manusia', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()

def is_headless():
    return os.environ.get('DISPLAY', '') == ''

if is_headless():
    print("🧠 Running in headless mode. Skipping GUI...")
    threading.Thread(target=run_camera, daemon=True).start()
    threading.Thread(target=poll_telegram_updates, daemon=True).start()

    # Keep the main thread alive so the background threads don't exit immediately
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopped by user")
else:
    print("🖥️ Display found. Launching GUI...")
    root = tk.Tk()
    root.title("YOLOv5 Human Detection Control")

    toggle_btn = ttk.Button(root, text="Turn ON Detection", command=toggle_detection)
    toggle_btn.pack(padx=20, pady=20)

    threading.Thread(target=run_camera, daemon=True).start()
    threading.Thread(target=poll_telegram_updates, daemon=True).start()

    root.mainloop()
