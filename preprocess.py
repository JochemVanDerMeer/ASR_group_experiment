# Code to preprocess image using different operators

import os
import cv2
import random
import numpy as np

from cv2 import resize

input_file_path = "/scratch/guchoadeassis/groupw/mlip_train_data/train_images"
output_file_path = "/scratch/guchoadeassis/groupw/preprocessed_mlip_train_data_warp_101_2"
mask_path = "/scratch/guchoadeassis/groupw/mlip_mask_data"
dirs = os.listdir(input_file_path)
mask_dirs = os.listdir(mask_path)
len_dirs = len(dirs)
len_masks = len(mask_dirs)

def add_mask(img):
    height, width, _ = img.shape
    mask_index = random.randint(0, len_masks-1)
    mask_img = cv2.imread(os.path.join(mask_path, mask_dirs[mask_index]), 0)
    mask_img_resized = cv2.resize(mask_img, (width, height))
    masked_img = np.array([[int(pixel != 76) for x, pixel in enumerate(line)] for y, line in enumerate(mask_img_resized)], dtype=np.uint8)
    masked_img = cv2.bitwise_and(img, img, mask=mask_img_bit)
    return masked_img

def resize_image(img, target_wh=512):
    height, width, _ = img.shape
    scale = 1
    if height >= width:
        scale = target_wh / height
    if width > height:
        scale = target_wh / width
    resizedImage = cv2.resize(img, (int(width * scale), int(height * scale)))
    new_height, new_width, _ = resizedImage.shape

    paddingTopBottom = int((target_wh - new_height) // 2)
    extraPaddingTopBottom = target_wh - (2 * paddingTopBottom) - new_height
    paddingLeftRight = int((target_wh - new_width) // 2)
    extraPaddingLeftRight = target_wh - (2 * paddingLeftRight) - new_width
        
    padded_image = cv2.copyMakeBorder(resizedImage, paddingTopBottom + extraPaddingTopBottom, paddingTopBottom, paddingLeftRight + extraPaddingLeftRight, paddingLeftRight, cv2.BORDER_REFLECT)
    return padded_image

def resize_image_warp(img, target_wh=512):
    resized_image = cv2.resize(img, (target_wh, target_wh))

    return resized_image

for i, file in enumerate(dirs):
    print(f'Preprocessing file {i}/{len_dirs}')
    if not os.path.isdir(os.path.join(input_file_path, file)):
        continue
    if not os.path.exists(f'{output_file_path}/{file}'):
        os.makedirs(f'{output_file_path}/{file}')
    for img in os.listdir(os.path.join(input_file_path, file)):
        imagepath = f'{input_file_path}/{file}/{img}'
        image = cv2.imread(imagepath)
        # image = add_mask(image)
        image = resize_image_warp(image)
        cv2.imwrite(f'{output_file_path}/{file}/{img}', image)