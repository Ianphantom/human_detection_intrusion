import cv2
import numpy as np
import onnxruntime as ort
import time

# === COCO Class Names ===
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# === Load ONNX Model ===
onnx_model_path = 'yolov5n.onnx'  # Replace with actual ONNX model path
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# === Helper: Preprocess frame for ONNX ===
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))  # HWC → CHW
    img_normalized = img_transposed.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    return img_input

# === Helper: Postprocess detections ===
def postprocess(outputs, orig_shape):
    boxes = []
    confidences = []
    class_ids = []

    for det in outputs[0][0]:  # Shape: (num_detections, 85)
        x1, y1, x2, y2, obj_conf, *class_scores = det
        class_id = np.argmax(class_scores)
        score = class_scores[class_id] * obj_conf

        if score > 0.4:  # Adjust threshold as needed
            w_ratio = orig_shape[1] / 640
            h_ratio = orig_shape[0] / 640

            x1 = int(x1 * w_ratio)
            y1 = int(y1 * h_ratio)
            x2 = int(x2 * w_ratio)
            y2 = int(y2 * h_ratio)

            boxes.append((x1, y1, x2, y2))
            confidences.append(score)
            class_ids.append(class_id)

    return boxes, confidences, class_ids

# === Start camera ===
cap = cv2.VideoCapture(0)  # 0 for default camera (adjust for Pi Cam if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    boxes, scores, class_ids = postprocess(outputs, frame.shape)

    for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, class_ids):
        label = f"{COCO_CLASSES[class_id]} {score:.2f}" if class_id < len(COCO_CLASSES) else f"Class {class_id}"
        color = (0, 255, 0) if class_id == 0 else (255, 100, 100)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('ONNX YOLOv5 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
