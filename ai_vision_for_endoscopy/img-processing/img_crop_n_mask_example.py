#!/usr/bin/env python3
# filename: img_crop_n_mask.py

from PIL import Image, ImageOps, ImageDraw
import os


def crop_mask_n_save_img_green(src_img_filepath,dst_img_filepath):
    # open
    pil_img = Image.open(src_img_filepath, 'r')
    # crop
    cropped_pil_img = crop_pil_img(pil_img)
    # mask
    cropped_n_masked_pil_img = mask_pil_img_green(cropped_pil_img)
    # save
    cropped_n_masked_pil_img.save(dst_img_filepath)

def crop_mask_n_save_img_black(src_img_filepath,dst_img_filepath):
    # open
    pil_img = Image.open(src_img_filepath, 'r')
    # crop
    cropped_pil_img = crop_pil_img(pil_img)
    # mask
    cropped_n_masked_pil_img = mask_pil_img_black(cropped_pil_img)
    # save
    cropped_n_masked_pil_img.save(dst_img_filepath)

def crop_n_save_img(src_img_filepath,dst_img_filepath):
    # open
    pil_img = Image.open(src_img_filepath, 'r')
    # crop
    cropped_pil_img = crop_pil_img(pil_img)
    # save
    cropped_pil_img.save(dst_img_filepath)

def crop_pil_img(pil_img):
    border = (32, 32, 32, 32) # left, top, right, bottom
    cropped_pil_img = ImageOps.crop(pil_img, border)
    return cropped_pil_img

def mask_pil_img_green(cropped_pil_img):
    # Create mask.
    #polygon_mask = Image.new('RGBA', (512,512), (0,0,0,255))
    polygon_mask = Image.new('RGBA', (512,512), (0,255,0,255)) # grn test mask
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

def mask_pil_img_black(cropped_pil_img):
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



crop_n_save_img('example.jpg', 'example_cropped.jpg')
crop_mask_n_save_img_green('example.jpg', 'example_crop_n_masked_green.jpg')
crop_mask_n_save_img_black('example.jpg', 'example_crop_n_masked_black.jpg')

