"""
# Author: ruben 
# Date: 8/7/22
# Project: MTLFramework
# File: image_normalization.py

Description: Compute image normalization over a defined dataset
"""
import os
import sys

from PIL import Image, ImageStat
import numpy as np


def get_custom_normalization(target):
    """
    Get normalization according to input train dataset
    :return: ((list) mean, (list) std) Normalization values mean and std
    """


    means = []
    stds = []

    for root, dirs, files in os.walk(target):
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


if __name__ == '__main__':
    args = sys.argv
    print(args)
    mean, std = get_custom_normalization(args[1])
    print(f'Mean: {mean} | Std: {std}')
