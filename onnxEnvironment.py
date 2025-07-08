import os
import cv2
import time
import json
import threading
import warnings
import tkinter as tk
from tkinter import ttk
from PIL import Image
import numpy as np
import requests
import onnxruntime as ort

# === Telegram Settings ===
TELEGRAM_BOT_TOKEN = '7585743264:AAGpvIaRIlJpEOLIfehCxew7F2ievdSgvZc'
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
telegram_chat_id = '1387459458'

ALERT_INTERVAL = 20 * 60  # 20 minutes
REMINDER_INTERVAL = 10    # seconds

last_alert_time = 0
alert_acknowledged = False
notification_feature_on = True
detection_on = False

# === ONNX Runtime Setup ===
session = ort.InferenceSession('yolov5n.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# === Preprocessing ===
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(img):
    img, _, _ = letterbox(img, new_shape=(640, 640))
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    return img[None]

# === Telegram Alert ===
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

# === Telegram Polling ===
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
                        print("✅ Alert acknowledged. Turning off detection.")
                        alert_acknowledged = True
                        last_alert_time = time.time()
                        detection_on = False
                        toggle_btn.config(text="Turn ON Detection")
                elif 'message' in update:
                    message = update['message']
                    chat_id = str(message['chat']['id'])
                    if chat_id != telegram_chat_id:
                        continue
                    text = message.get('text', '').strip().upper()
                    if text == "TURN_ON":
                        detection_on = True
                        last_alert_time = 0
                        alert_acknowledged = False
                        toggle_btn.config(text="Turn OFF Detection")
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
                            'chat_id': telegram_chat_id,
                            'text': "✅ Detection is now ON."
                        })
                    elif text == "TURN_OFF":
                        detection_on = False
                        toggle_btn.config(text="Turn ON Detection")
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
                            'chat_id': telegram_chat_id,
                            'text': "❌ Detection is now OFF."
                        })
        except Exception as e:
            print("Polling error:", e)
        time.sleep(1)

# === Toggle Button ===
def toggle_detection():
    global detection_on, last_alert_time, alert_acknowledged
    detection_on = not detection_on
    last_alert_time = 0
    alert_acknowledged = False
    toggle_btn.config(text="Turn OFF Detection" if detection_on else "Turn ON Detection")

# === Camera Thread ===
def run_camera():
    global last_alert_time, alert_acknowledged, detection_on
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if detection_on:
            input_tensor = preprocess(frame)
            outputs = session.run(None, {input_name: input_tensor})
            preds = outputs[0][0]
            for pred in preds:
                if pred[4] < 0.25:
                    continue
                cls = np.argmax(pred[5:])
                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, pred[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    current_time = time.time()
                    if notification_feature_on and not alert_acknowledged and (current_time - last_alert_time > REMINDER_INTERVAL):
                        img_path = 'human_detected.jpg'
                        cv2.imwrite(img_path, frame)
                        send_telegram_alert(img_path)
                        last_alert_time = current_time
                        break
                    elif alert_acknowledged and (current_time - last_alert_time > ALERT_INTERVAL):
                        print("⏰ Reminder time passed, turning detection back ON.")
                        alert_acknowledged = False
                        detection_on = True
                        toggle_btn.config(text="Turn OFF Detection")
                    break

        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-5)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow('YOLOv5 ONNX - Human Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# === GUI Setup ===
root = tk.Tk()
root.title("YOLOv5 ONNX Human Detection")

toggle_btn = ttk.Button(root, text="Turn ON Detection", command=toggle_detection)
toggle_btn.pack(padx=20, pady=20)

# === Threads ===
threading.Thread(target=run_camera, daemon=True).start()
threading.Thread(target=poll_telegram_updates, daemon=True).start()

root.mainloop()
