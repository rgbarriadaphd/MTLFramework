"""
# Author: ruben 
# Date: 25/10/22
# Project: MTLFramework
# File: regenerate_folds.py

Description: "Enter feature description here"
"""
import json
import os
from PIL import Image
import numpy as np
import cv2

org_negatives = '/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents/Controles'
org_positives = '/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents/original_retina_images'
assert os.path.exists(org_negatives) and os.path.exists(org_positives)

base_output = '/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents/cnn_datasets'
baseline_output = os.path.join(base_output, 'baseline')
clahe_output = os.path.join(base_output, 'clahe')

transformation = {
    # '2196x1958': ['resize', 'crop_110'],
    '2196x1958': ['resize', 'crop_119_0'],
    # '720x576': ['resize', 'crop_20'],
    '720x576': ['resize', 'crop_72_0'],
    '2592x1944': ['resize', 'crop_324'],
    # '2124x2056': ['resize', 'crop_34'],
    '2124x2056': ['resize', 'crop_34_0'],
    '1792x1184': ['resize', 'crop_438_290_60'],
    '3744x3744': ['resize', 'crop_110'],
    '1024x768': ['resize', 'crop_168_40'],
    '2048x1536': ['resize', 'crop_330_80'],
    '1956x1934': ['resize', 'crop_56_34_34']
}


def create_output_structure():
    if os.path.exists(base_output):
        print("Output structure already created")
    else:
        print("Create output structure")
        os.makedirs(base_output)

        for elem in ['baseline', 'clahe']:
            os.makedirs(os.path.join(base_output, elem))
            for i in range(1, 11):
                os.makedirs(os.path.join(base_output, elem, f'outer_fold_{i}'))
                for j in range(1, 6):
                    os.makedirs(os.path.join(base_output, elem, f'outer_fold_{i}', f'inner_fold_{j}'))
                    os.makedirs(os.path.join(base_output, elem, f'outer_fold_{i}', f'inner_fold_{j}', 'CEN'))
                    os.makedirs(os.path.join(base_output, elem, f'outer_fold_{i}', f'inner_fold_{j}', 'CEP'))


def load_json():
    with open('/home/ruben/PycharmProjects/MTLFramework/input/CE/organization.json', 'r') as f:
        json_dict = json.load(f)
    return json_dict


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


def transform_image(image_path, partial_out_path):
    # load
    org = Image.open(image_path)
    width, height = org.size

    operations = transformation[f'{width}x{height}']
    assert f'{width}x{height}' in list(transformation.keys())

    # Crop
    cropped = crop(org, operations[1])
    cw, ch = cropped.size
    # if cw != ch:
    #     if abs(cw - ch) > 12:
    #         print(f'{width}x{height}', org.size, abs(cw - ch), image_path)

    # clahe
    clahe = clahe_image(cropped)

    # resize
    #baseline_im = resize(cropped)
    clahe_im = resize(clahe)

    # Save
    # baseline_im.save(os.path.join(baseline_output, partial_out_path))
    clahe_im.save(os.path.join(clahe_output, partial_out_path))


def clahe_image(image):
    bgr = np.array(image)
    # Convert RGB to BGR
    bgr = bgr[:, :, ::-1].copy()
    # bgr = cv2.imread(image_path)

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
    return im_pil

def convert_images(folds_organization):
    for outer_fold in folds_organization.keys():
        for inner_fold in folds_organization[outer_fold].keys():

            # Iterate positive images
            positive_images_list = folds_organization[outer_fold][inner_fold]['CEP']
            for pi in positive_images_list:
                image_name = pi
                if pi not in os.listdir(org_positives):
                    reversed_name_list = pi.split('_')
                    reversed_name_list[0], reversed_name_list[1] = reversed_name_list[1], reversed_name_list[0]
                    reversed_name = "_".join(reversed_name_list)
                    assert reversed_name in os.listdir(org_positives), f'{pi}, {reversed_name}'
                    image_name = reversed_name

                partial_out_path = os.path.join(f'outer_fold_{outer_fold}', f'inner_fold_{inner_fold}', 'CEP', image_name)
                transform_image(os.path.join(org_positives, image_name), partial_out_path)

            # Iterate negative images
            negative_images_list = folds_organization[outer_fold][inner_fold]['CEN']
            for ni in negative_images_list:
                image_name = ni
                if ni not in os.listdir(org_negatives):
                    reversed_name_list = ni.split('_')
                    reversed_name_list[0], reversed_name_list[1] = reversed_name_list[1], reversed_name_list[0]
                    reversed_name = "_".join(reversed_name_list)
                    assert reversed_name in os.listdir(org_negatives), f'{ni}, {reversed_name}'
                    image_name = reversed_name

                partial_out_path = os.path.join(f'outer_fold_{outer_fold}', f'inner_fold_{inner_fold}', 'CEN', image_name)
                transform_image(os.path.join(org_negatives, image_name), partial_out_path)


    # iterate organization
    # find image


if __name__ == '__main__':
    # Generate output structure
    create_output_structure()

    # load json structure
    folds_organization = load_json()

    convert_images(folds_organization)
