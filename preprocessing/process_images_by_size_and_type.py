"""
# Author: ruben 
# Date: 21/9/22
# Project: CardioEvents
# File: process_images_by_size_and_type.py

Description: Process imagen depending on morphology.
"""
import os
import shutil

from PIL import Image

BASE_FOLDER = '/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents'
ORG_SIZE_FOLDER = os.path.join(BASE_FOLDER, 'filtered_by_size')
TARGET_TEST = os.path.join(BASE_FOLDER, 'test_images')

transformation = {
    '2196x1958': ['resize', 'crop_110'],
    '720x576': ['resize', 'crop_20'],
    '2592x1944': ['resize', 'crop_324'],
    '2124x2056': ['resize', 'crop_34'],
    '1792x1184': ['resize', 'crop_438_290_60'],
    '3744x3744': ['resize', 'crop_110'],
    '1024x768': ['resize', 'crop_168_40'],
    '2048x1536': ['resize', 'crop_330_80'], #
    '1956x1934': ['resize', 'crop_56_34_34']
}


def crop(image, parameters):
    width, height = image.size
    params = parameters.split('_')
    params.pop(0)
    params = [int(p) for p in params]

    if len(params) == 1:
        left = params[0]
        right = width - left
        top = left
        bottom = height - top
    elif len(params) == 2:
        left = params[0]
        right = width - left
        top = params[1]
        bottom = height - top
    elif len(params) == 3:
        left = params[0]
        right = width - params[1]
        top = params[2]
        bottom = height - top

    return image.crop((left, top, right, bottom))


def resize(image):
    return image.resize((224, 224))


def main():
    # creating a object
    for size_folder in os.listdir(ORG_SIZE_FOLDER):
        operations = transformation[size_folder]

        for image_path in os.listdir(os.path.join(ORG_SIZE_FOLDER, size_folder)):
            im = Image.open(os.path.join(ORG_SIZE_FOLDER, size_folder, image_path))

            # crop
            im = crop(im, operations[1])
            # resize
            im = resize(im)

            im.save(os.path.join(TARGET_TEST, image_path))


def main2():
    im = Image.open('/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents/filtered_by_size/2048x1536/17188_1045372_1_F91096725017188_20180306_140849_Color_R_03.jpg')
    width, height = im.size
    # im.show()
    print(im.size)
    left = 330
    right = width - left
    top = 80
    bottom = height - top


    crop = im.crop((left, top, right, bottom))
    print(crop.size)
    crop.show()


if __name__ == '__main__':
    main()
