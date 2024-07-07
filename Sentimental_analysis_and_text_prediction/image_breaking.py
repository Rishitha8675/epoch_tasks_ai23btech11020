import numpy as np
import pandas as pd
import cv2
import os

for x in range(1, 7):
    image_path = f"/home/rishitha/epoch_tasks/sentimental_analysis/target_images/line_{x}.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    i = 0
    j = 0
    letter_number=0
    os.mkdir(f"text_sentence_{x}")
    while True:
        if j + 28 < img.shape[0]:
            if i + 28 >= img.shape[1]:
                img_crop = img[j:j + 28, i:img.shape[1]]
                j += 28
                i = 0
            else:
                img_crop = img[j:j + 28, i:i + 28]
                i += 28
        else:
            if i + 28 > img.shape[1]:
                img_crop = img[j:img.shape[0], i:img.shape[1]]
                
                break
            else:
                img_crop = img[j:img.shape[0], i:i + 28]
                i += 28
        cv2.imwrite(f'text_sentence_{x}/file_name{letter_number}.png', img_crop)
        letter_number += 1

