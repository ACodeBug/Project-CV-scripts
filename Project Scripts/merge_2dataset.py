import os
import shutil
from pathlib import Path

# Path

first_image_dir = r"Merge1_dataset\images\train"
first_label_dir = r"Merge1_dataset\labels\train"

second_image_dir = r"Dataset_merged_StreetAugm_Animal\images\train"
second_label_dir = r"Dataset_merged_StreetAugm_Animal\labels\train"

target_image_dir = r"Result_dataset\images\train"
target_label_dir = r"Result_dataset\labels\train"


# Creating the merge folder
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

# New name for merged images in some kind logic
def copy_with_new_name(src_img, src_lbl, target_img_dir, target_lbl_dir, prefix):
    base_name = os.path.splitext(os.path.basename(src_img))[0]
    ext = os.path.splitext(src_img)[1]  # .jpg or .png
    new_base = f"{prefix}_{base_name}"
    new_img = os.path.join(target_img_dir, f"{new_base}{ext}")
    new_lbl = os.path.join(target_lbl_dir, f"{new_base}.txt")
    shutil.copy(src_img, new_img)
    shutil.copy(src_lbl, new_lbl)

# 1. First Source (Dataset)
for img_file in Path(first_image_dir).glob("*"):
    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        lbl_file = Path(first_label_dir) / (img_file.stem + ".txt")
        if lbl_file.exists():
            copy_with_new_name(str(img_file), str(lbl_file), target_image_dir, target_label_dir, "first")
        else:
            print("⚠️ Label fehlt für:", img_file.name)

# 2. Second Source (Dataset)
for img_file in Path(second_image_dir).glob("*.jpg"):
    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        lbl_file = Path(second_label_dir) / (img_file.stem + ".txt")
        if lbl_file.exists():
            copy_with_new_name(str(img_file), str(lbl_file), target_image_dir, target_label_dir, "second")
        else:
            print("⚠️ Label fehlt für:", img_file.name)

print("✅ Merged Dataset saved under:", target_image_dir)
