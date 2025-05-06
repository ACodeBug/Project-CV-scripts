# Not ready, just for understanding and for idea 
from ultralytics import YOLO
import cv2
import os

CONF_THRESHOLD = 0.3
NEW_CLASS_NAME = "unknown"
MODEL_PATH = "yolov8n.pt"  # oder dein trainiertes Modell

model = YOLO(MODEL_PATH)
image_path = "images/example.jpg"
img = cv2.imread(image_path)

# Vorhersage
results = model(image_path, conf=0.01)  # bewusst niedriger conf für Analyse
result = results[0]

unknown_boxes = []

for box in result.boxes:
    conf = float(box.conf)
    cls = int(box.cls)
    if conf < CONF_THRESHOLD:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        unknown_boxes.append(xyxy)
        # Zeichne die Box ins Bild
        cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 255), 2)
        cv2.putText(img, NEW_CLASS_NAME, tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite("runs/unknown_detected.jpg", img)

# Speichere als YOLO-Label (falls gewünscht)
if unknown_boxes:
    h, w = img.shape[:2]
    label_path = "runs/example.txt"
    with open(label_path, "w") as f:
        for xyxy in unknown_boxes:
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{YOUR_NEW_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
