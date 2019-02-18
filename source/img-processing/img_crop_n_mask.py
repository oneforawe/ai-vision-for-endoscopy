#!/usr/bin/env python3
# filename: img_crop_n_mask.py

from PIL import Image, ImageOps, ImageDraw
import os

# Instructions:
# 1) Copy pre-processed images (under input-data/1-pre-processed/)
#    into a "processed" folder (under input-data/2-processed/).
# 2) Set the images `path_base` (folder where the images are located) below.
# 3) Execute this code to transform (crop-and-mask) the images in-place.

# Note:
# This could be done more efficiently by not re-creating the mask for each img.


def crop_mask_n_save_img(src_img_filepath,dst_img_filepath):
    # open
    pil_img = Image.open(src_img_filepath, 'r')
    # crop
    cropped_pil_img = crop_pil_img(pil_img)
    # mask
    cropped_n_masked_pil_img = mask_pil_img(cropped_pil_img)
    # save
    cropped_n_masked_pil_img.save(dst_img_filepath)

def crop_pil_img(pil_img):
    border = (32, 32, 32, 32) # left, top, right, bottom
    cropped_pil_img = ImageOps.crop(pil_img, border)
    return cropped_pil_img

def mask_pil_img(cropped_pil_img):
    # Create mask.
    polygon_mask = Image.new('RGBA', (512,512), (0,0,0,255))
    #polygon_mask = Image.new('RGBA', (512,512), (0,255,0,255)) # grn test mask
    pdraw = ImageDraw.Draw(polygon_mask)
    pdraw.polygon( [ (0,125),(50,50),(125,0),
                     (512-125,0),(512-50,50),(512,125),
                     (512,512-125),(512-50,512-50),(512-125,512),
                     (125,512),(50,512-50),(0,512-125) ],
                   fill=(0,0,0,0) )
    # Apply mask.
    cropped_pil_img.paste(polygon_mask, mask=polygon_mask)
    cropped_n_masked_pil_img = cropped_pil_img
    return cropped_n_masked_pil_img


path_base = 'put-your-image-path-here'
#path_base = '../../input-data/2-processed/F/'
#path_base = '../../input-data/2-processed/E/'
#path_base = '../../input-data/2-processed/D/'
#path_base = '../../input-data/2-processed/C/'
#path_base = '../../input-data/2-processed/test/'

for dirPath, subDirs, files in os.walk(path_base):
    for file in files:
        if file.endswith(".jpg"):
            crop_mask_n_save_img( os.path.join(dirPath, file),
                                  os.path.join(dirPath, file) )


