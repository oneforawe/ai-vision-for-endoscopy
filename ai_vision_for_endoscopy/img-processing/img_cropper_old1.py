#!/usr/bin/env python3
# filename: img_cropper.py

#import PIL
from PIL import Image, ImageOps

def crop_pil_img(pil_img):
    border = (32, 32, 32, 32) # left, top, right, bottom
    cropped_pil_img = ImageOps.crop(pil_img, border)
    return cropped_pil_img

def crop_n_save_img(img_filepath,cropped_filepath):
    pil_img = Image.open(img_filepath, 'r')
    cropped_pil_img = crop_pil_img(pil_img)
    cropped_pil_img.save(cropped_filepath)

#def crop_n_save_all_imgs_in_folder(folder_path):
#
#    crop_n_save_img(


filenames = ['1376916 20 Dec 18_1 140.jpg',
             '2488865 12 Dec 18_1 0207.jpg',
             '2488865 12 Dec 18_1 1591.jpg',
             '2805206 01 Dec 18_1 08.jpg',
             '11043364 28 Nov 18_1 0001.jpg',
             '11043364 28 Nov 18_1 1606.jpg']
"""
i = 0
for filename in filenames:
    crop_n_save_img(filename,f'{i:02d}.jpg')
    i += 1
"""

#crop_n_save_img(filenames[0],f'00.jpg')


src_path_base = '../../input-data/1-pre-processed/B/'
dst_path_base = '../../input-data/2-processed/B/'
i = 0
for dirPath, subDirs, files in os.walk(src_path_base+'Normal/'):
        for file in files:
            if file.endswith(".jpg"):
                crop_n_save_img(file,dst_path_base+f'Normal/{i:02d}.jpg')
i = 0
for dirPath, subDirs, files in os.walk(src_path_base+'Abnormal/'):
        for file in files:
            if file.endswith(".jpg"):
                crop_n_save_img(file,dst_path_base+f'Abnormal/{i:02d}.jpg')




