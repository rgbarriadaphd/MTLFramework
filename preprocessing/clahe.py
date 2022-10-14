"""
# Author: ruben
# Date: 27/9/22
# Project: MTLFramework
# File: clahe.py

Description: "Enter feature description here"
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageStat

image_path = '/home/ruben/Documentos/Doctorado/datasets/ScoreCalcico/Original/CACs mas 400/p5 OIc.jpg'
# image_path = '/home/ruben/Documentos/Doctorado/datasets/ScoreCalcico/Original/CACs mas 400/p69 ODc.jpg'

def main():
    assert os.path.exists(image_path)
    # Reading the image from the present directory
    image = cv2.imread(image_path)
    img = Image.open(image_path)
    # img.show()

    # B/N
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30
    im_pil = Image.fromarray(final_img)
    # im_pil.show()

    # COLOR
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes = list(lab_planes)
    clahe = cv2.createCLAHE(clipLimit=5.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab_planes[1] = clahe.apply(lab_planes[1])
    lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    im_pil = Image.fromarray(bgr)
    # im_pil.show()


    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    # final_img = clahe.apply(lab)
    # im_pil = Image.fromarray(final_img)
    # im_pil.show()

    # We first create a CLAHE model based on OpenCV
    # clipLimit defines threshold to limit the contrast in case of noise in our image
    # tileGridSize defines the area size in which local equalization will be performed
    colorimage = cv2.imread(image_path)
    clahe_model = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(colorimage[:, :, 0])
    colorimage_g = clahe_model.apply(colorimage[:, :, 1])
    colorimage_r = clahe_model.apply(colorimage[:, :, 2])
    # Next we stack our equalized channels back into a single image
    colorimage_clahe = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    im_pil = Image.fromarray(colorimage_clahe)
    im_pil.show()

    # img = cv2.imread(image_path)
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    # # h = clahe.apply(h)
    # v = clahe.apply(v)
    # # s = clahe.apply(s)
    # hsv_img = np.dstack((h, s, v))
    # rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    # im_pil = Image.fromarray(rgb)
    # im_pil.show()


if __name__ == '__main__':
    main()
