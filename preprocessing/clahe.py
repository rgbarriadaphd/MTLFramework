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

import time

def b_clahe_nlm_de(image_path):
    t0 = time.time()
    bgr = cv2.imread(image_path)

    bgr = cv2.bilateralFilter(bgr, 3, 3, 2)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5)
    lab = cv2.merge((clahe.apply(lab_planes[0]), lab_planes[1], lab_planes[2]))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 3, 9)
    bgr = cv2.detailEnhance(bgr, sigma_s=10, sigma_r=0.15)


    color_coverted = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


    im_pil = Image.fromarray(color_coverted)
    cropped = im_pil.resize((224, 224))
    print(f'Elapsed: {time.time() - t0} s')
    cropped.show()
    # im_pil.show()

    # im_org = Image.open(image_path)
    # im_org.show()



def clahe(input_image):
    assert os.path.exists(input_image)
    print(f'processing: {input_image}')
    bgr = cv2.imread(input_image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    lab = cv2.merge((clahe.apply(lab_planes[0]), lab_planes[1], lab_planes[2]))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
    im_pil = Image.fromarray(bgr)
    im_pil.show()
    # im_pil.save(input_image)
    # cv2.imwrite(input_image, bgr)

# def convert_images():
#     for root, dirs, files in os.walk('/home/ruben/PycharmProjects/MTLFramework/input/CE/outer_folds_clahe'):
#         for file in files:
#             file_path = os.path.join(root, file)
#             clahe(file_path)

if __name__ == '__main__':
    b_clahe_nlm_de(image_path)

