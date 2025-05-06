from rembg import remove
from PIL import Image
import os

# Usage: for structured folders. Remove backgrounds in "dataset" and from all folders below
input_folder = 'Dataset_Normal_Streets/dataset50/Animals/dataset'
output_folder = 'C:/Users/justm/Desktop/ausgabe_bilder'
os.makedirs(output_folder, exist_ok=True)

# Image types
bildformate = ('.jpg', '.jpeg', '.png', '.webp')

def crop_transparency(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        return img.crop(bbox)
    return img

# All folders under main folder loop
for root, dirs, files in os.walk(input_folder):
    for dateiname in files:
        if dateiname.lower().endswith(bildformate):
            path = os.path.join(root, dateiname)

            # Relative Path getter
            rel_path = os.path.relpath(path, input_folder)
            ziel_pfad = os.path.join(output_folder, os.path.splitext(rel_path)[0] + '_no_bg_cropped.png')

            # Goal folder (inkl. underfolder)
            os.makedirs(os.path.dirname(ziel_pfad), exist_ok=True)

            print(f"Edited: {rel_path}...")

            image = Image.open(path)
            image_without_bg = remove(image)
            image_crop = crop_transparency(image_without_bg)
            image_crop.save(ziel_pfad)

print("Ready! All Images have to be edited und cropped.")
