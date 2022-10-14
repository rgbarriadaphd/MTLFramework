"""
# Author: ruben 
# Date: 28/9/22
# Project: MTLFramework
# File: normalization.py

Description: computes normalization for a specific fold and stores it in a csv to be loaded when used.
"""
import os
import json
from PIL import Image, ImageStat
import numpy as np

BASE_PATH = '/home/ruben/PycharmProjects/MTLFramework/input/CAC'


def get_custom_normalization(path):
    """
    Get normalization according to input train dataset
    :param: path (string) path of the images to get the normalization values
    :return: ((list) mean, (list) std) Normalization values mean and std
    """
    means = []
    stds = []

    for root, dirs, files in os.walk(path):
        for file in files:
            image_path = os.path.join(root, file)
            assert os.path.exists(image_path)
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                stat = ImageStat.Stat(img)
                local_mean = [stat.mean[0] / 255., stat.mean[1] / 255., stat.mean[2] / 255.]
                local_std = [stat.stddev[0] / 255., stat.stddev[1] / 255., stat.stddev[2] / 255.]
                means.append(local_mean)
                stds.append(local_std)

    return list(np.array(means).mean(axis=0)), list(np.array(stds).mean(axis=0))


def main():
    outer_folds = os.path.join(BASE_PATH, 'outer_folds')

    parameters = {}
    for i in range(1, 10):
        parameters[i] = {}
        for j in range(1, 5):
            src = os.path.join(outer_folds, f'outer_fold_{i}', f'inner_fold_{j}')
            assert os.path.exists(src)
            m, s = get_custom_normalization(src)
            parameters[i][j] = {'mean': m, 'std': s}

    with open(os.path.join(BASE_PATH, 'normalization.json'), 'w') as f:
        json.dump(parameters, f)


if __name__ == '__main__':
    main()
