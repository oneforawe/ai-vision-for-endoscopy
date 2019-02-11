#!/usr/bin/env python3
# filename: img_cropper.py

from PIL import Image, ImageOps

def crop_pil_img(pil_img):
    border = (32, 32, 32, 32) # left, top, right, bottom
    cropped_pil_img = ImageOps.crop(pil_img, border)
    return cropped_pil_img

def crop_n_save_img(img_filepath,cropped_filepath):
    pil_img = Image.open(img_filepath, 'r')
    cropped_pil_img = crop_pil_img(pil_img)
    cropped_pil_img.save(cropped_filepath)

def crop_n_save_all_imgs_in_folder(folder_path):
#   Need to have a robust naming system across many folders.
#   crop_n_save_img()

