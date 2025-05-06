import os
import cv2
import random
import uuid
from PIL import Image
import shutil

# === Eingabepfade (ANPASSEN WENN NÖTIG) ===
streets_images_dir = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\obj_Train_data\images\train"
streets_labels_dir = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\obj_Train_data\labels\train"

animals_images_dir = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\dataset50\Structured Animals NoBg\images\train"
animals_labels_dir = r"C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\dataset50\Structured Animals NoBg\labels\train"

# === Ausgabeordner ===
output_images_dir = r"Dataset Street_Animals/images/train"
output_labels_dir = r"Dataset Street_Animals/labels/train"
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# === YOLO-Klassenkonfiguration ===
animal_class_id = 188  # Generische Klasse "Animal"

# === Lade Tiere und Labels ===
animal_files = [f for f in os.listdir(animals_images_dir) if f.lower().endswith(".png")]
street_files = [f for f in os.listdir(streets_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
selected_streets = random.sample(street_files, int(len(street_files) * 0.2))  # 20 % der Straßenbilder

def yolo_to_bbox(yolo_line, img_width, img_height):
    parts = yolo_line.strip().split()
    class_id, cx, cy, w, h = map(float, parts)
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return int(class_id), int(x1), int(y1), int(x2), int(y2)

def bbox_to_yolo(class_id, x1, y1, x2, y2, img_width, img_height):
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

for street_file in selected_streets:
    street_path = os.path.join(streets_images_dir, street_file)
    street_img = Image.open(street_path).convert("RGBA")
    img_width, img_height = street_img.size

    # Street-Labels read
    street_label_path = os.path.join(streets_labels_dir, os.path.splitext(street_file)[0] + ".txt")
    with open(street_label_path, "r") as f:
        street_labels = f.readlines()

    all_labels = street_labels[:]

    # how many times "animal" has to appier in the street image (not on position depend!)
    num_animals = random.randint(1, 3)
    selected_animals = random.sample(animal_files, num_animals)

    for animal_file in selected_animals:
        animal_img_path = os.path.join(animals_images_dir, animal_file)
        animal_label_path = os.path.join(animals_labels_dir, os.path.splitext(animal_file)[0] + ".txt")

        animal_img = Image.open(animal_img_path).convert("RGBA")
        aw, ah = animal_img.size

        # How big the "sticker" on the image hast to be
        scale = 300 / max(aw, ah)
        new_size = (int(aw * scale), int(ah * scale))
        animal_img = animal_img.resize(new_size, Image.LANCZOS)
        aw, ah = new_size

        # Random Positioning,
        max_x = max(0, img_width - aw)
        max_y = max(0, img_height - ah)
        x_offset = random.randint(0, max_x)
        y_offset = random.randint(0, max_y)

        # Merging
        street_img.paste(animal_img, (x_offset, y_offset), animal_img)

        with open(animal_label_path, "r") as f:
            for line in f:
                class_id, x1, y1, x2, y2 = yolo_to_bbox(line, *new_size)
                # move to street image
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset

                # verification to stay inside of the image borders by rotation as example
                x1 = min(max(0, x1), img_width - 1)
                y1 = min(max(0, y1), img_height - 1)
                x2 = min(max(1, x2), img_width)
                y2 = min(max(1, y2), img_height)

                all_labels.append(bbox_to_yolo(class_id, x1, y1, x2, y2, img_width, img_height) + "\n")
                all_labels.append(bbox_to_yolo(animal_class_id, x1, y1, x2, y2, img_width, img_height) + "\n")

    # New UUID für Datei
    base_name = os.path.splitext(street_file)[0]
    new_name = f"{base_name}_{uuid.uuid4().hex[:8]}"
    out_image_path = os.path.join(output_images_dir, new_name + ".jpg")
    out_label_path = os.path.join(output_labels_dir, new_name + ".txt")

    street_img = street_img.convert("RGB")
    street_img.save(out_image_path, "JPEG")

    with open(out_label_path, "w") as f:
        f.writelines(all_labels)

print("✅ Ready: augmented images with animals saved in 'Dataset Street_Animals'")
