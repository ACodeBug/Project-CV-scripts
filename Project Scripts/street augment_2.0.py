import os
import cv2
import random
import albumentations as A
from glob import glob
from tqdm import tqdm
import shutil

# Original source
images_dir = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\obj_Train_data\images\train"
labels_dir = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\obj_Train_data\labels\train"

# Goalfile
augm_root = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Streets_Augm"
augm_images = os.path.join(augm_root, "images", "train")
augm_labels = os.path.join(augm_root, "labels", "train")
os.makedirs(augm_images, exist_ok=True)
os.makedirs(augm_labels, exist_ok=True)

# YOLO-konforme Augmentpipeline with BoundingBox-label adoptation
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.8),
    A.MotionBlur(blur_limit=11, p=0.5),
    A.RandomGamma(p=0.6),
    A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.7),
    A.RandomSizedCrop(min_max_height=(400, 720), size=(720, 1280), p=0.6),
],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

# Helpfunction to read of YOLO labels
def read_yolo_labels(label_path):
    boxes = []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append(int(parts[0]))
                boxes.append([float(x) for x in parts[1:]])
    return boxes, labels

# Helpfunction to write the labels
def write_yolo_labels(path, labels, boxes):
    with open(path, 'w') as f:
        for cls, box in zip(labels, boxes):
            f.write(f"{cls} {' '.join(f'{b:.6f}' for b in box)}\n")

# All images pathes getting
image_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(os.path.join(images_dir, "*.png"))

# images augmented and save
for img_path in tqdm(image_paths, desc="Augmentiere Bilder"):
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    label_path = os.path.join(labels_dir, name + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    boxes, labels = read_yolo_labels(label_path)

    try:
        transformed = transform(image=image, bboxes=boxes, class_labels=labels)
        aug_image = transformed['image']
        aug_boxes = transformed['bboxes']
        aug_labels = transformed['class_labels']

        if len(aug_boxes) == 0:
            continue  # checkt if still some Box exist

        # save
        aug_filename = f"{name}_aug.jpg"
        cv2.imwrite(os.path.join(augm_images, aug_filename), aug_image)
        write_yolo_labels(os.path.join(augm_labels, name + "_aug.txt"), aug_labels, aug_boxes)

    except Exception as e:
        print(f"Fehler bei {name}: {e}")
