#!/usr/bin/env python3
# filename: img_cropper.py

from PIL import Image, ImageOps
import os

def crop_pil_img(pil_img):
    border = (32, 32, 32, 32) # left, top, right, bottom
    cropped_pil_img = ImageOps.crop(pil_img, border)
    return cropped_pil_img

def crop_n_save_img(img_filepath,cropped_filepath):
    pil_img = Image.open(img_filepath, 'r')
    cropped_pil_img = crop_pil_img(pil_img)
    cropped_pil_img.save(cropped_filepath)

#def crop_n_save_all_imgs_in_folder(folder_path):
#   In general, need to have a robust naming system across many folders.
#   crop_n_save_img()


src_path_base = '../../input-data/1-pre-processed/B/'
dst_path_base = '../../input-data/2-processed/B/'
i = 0
for dirPath, subDirs, files in os.walk(src_path_base+'Normal/'):
        for file in files:
            if file.endswith(".jpg"):
                i += 1
                crop_n_save_img( os.path.join(dirPath, file) ,
                                 dst_path_base+f'Normal/{i:02d}.jpg' )
i = 0
for dirPath, subDirs, files in os.walk(src_path_base+'Abnormal/'):
        for file in files:
            if file.endswith(".jpg"):
                i += 1
                crop_n_save_img( os.path.join(dirPath, file) ,
                                 dst_path_base+f'Abnormal/{i:02d}.jpg' )

