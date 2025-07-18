import cv2
import numpy as np
import os
import warnings
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image
import time
import requests
import json
import onnxruntime as ort

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

# === Load YOLOv5 ONNX Model ===
onnx_model_path = 'yolov5n.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

def preprocess(frame):
    img = cv2.resize(frame, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, frame, conf_thres=0.4):
    boxes, scores, classes = [], [], []

    predictions = output[0]  # [1, num_boxes, 85]
    for pred in predictions[0]:
        conf = pred[4]
        if conf < conf_thres:
            continue
        cls_scores = pred[5:]
        cls_id = np.argmax(cls_scores)
        score = cls_scores[cls_id] * conf
        if cls_id == 0 and score > conf_thres:
            x, y, w, h = pred[0:4]
            x1 = int((x - w / 2) * frame.shape[1] / 640)
            y1 = int((y - h / 2) * frame.shape[0] / 640)
            x2 = int((x + w / 2) * frame.shape[1] / 640)
            y2 = int((y + h / 2) * frame.shape[0] / 640)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            classes.append(cls_id)
    return boxes, scores, classes

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
                        print("🔔 Detection turned ON via Telegram")
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
                            'chat_id': telegram_chat_id,
                            'text': "✅ Detection is now ON."
                        })
                        toggle_btn.config(text="Turn OFF Detection")
                    elif text == "TURN_OFF":
                        detection_on = False
                        print("🛑 Detection turned OFF via Telegram")
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
                            'chat_id': telegram_chat_id,
                            'text': "❌ Detection is now OFF."
                        })
                        toggle_btn.config(text="Turn ON Detection")
        except Exception as e:
            print("Polling error:", e)
        time.sleep(1)

def run_camera():
    global last_alert_time, alert_acknowledged, detection_on, detection_pause_until
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if detection_on and current_time >= detection_pause_until:
            img_input = preprocess(frame)
            ort_inputs = {ort_session.get_inputs()[0].name: img_input}
            ort_outs = ort_session.run(None, ort_inputs)
            boxes, scores, classes = postprocess(ort_outs, frame)

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if notification_feature_on and not alert_acknowledged and (current_time - last_alert_time > REMINDER_INTERVAL):
                    img_path = 'human_detected.jpg'
                    cv2.imwrite(img_path, frame)
                    send_telegram_alert(img_path)
                    last_alert_time = current_time
                    detection_pause_until = current_time + 5
                    print("⏭️ Skipping detection for next 50 seconds.")
                    break
                elif notification_feature_on and alert_acknowledged and (current_time - last_alert_time > ALERT_INTERVAL):
                    print("⏰ Reminder time passed, turning detection back ON.")
                    alert_acknowledged = False
                    detection_on = True
                    toggle_btn.config(text="Turn OFF Detection")
                break

        fps = 1 / (time.time() - start_time + 1e-5)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('YOLOv5 ONNX - Human Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()

def is_headless():
    return os.environ.get('DISPLAY', '') == ''

# === GUI Setup or Headless ===
# if is_headless():
#     print("🧠 Running in headless mode. Skipping GUI...")
#     threading.Thread(target=run_camera, daemon=True).start()
#     threading.Thread(target=poll_telegram_updates, daemon=True).start()
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("🛑 Stopped by user")
# else:
print("🖥️ Display found. Launching GUI...")
root = tk.Tk()
root.title("ONNX YOLOv5 Human Detection Control")
toggle_btn = ttk.Button(root, text="Turn ON Detection", command=toggle_detection)
toggle_btn.pack(padx=20, pady=20)
threading.Thread(target=run_camera, daemon=True).start()
threading.Thread(target=poll_telegram_updates, daemon=True).start()
root.mainloop()
