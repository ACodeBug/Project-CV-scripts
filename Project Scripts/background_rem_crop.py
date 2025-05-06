from rembg import remove
from PIL import Image
import os

# Input of your image for testing the background removing
input_path = 'b5803098656346a.png'
output_path = 'output_cropped_2.png'

input_image = Image.open(input_path)
output_image = remove(input_image)

# cropping the transparent pixels to the "not transparent" pixels
def crop_to_content(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        return img.crop(bbox)
    return img  # If the whole image is Transparent 

output_cropped = crop_to_content(output_image)

output_cropped.save(output_path)

print(f"Images saved under: {output_path}")
