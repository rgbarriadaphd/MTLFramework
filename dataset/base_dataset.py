"""
# Author: ruben 
# Date: 14/6/22
# Project: MTLFramework
# File: base_dataset.py

Description: Class that implements Base dataset (CAC scoring)
"""

import logging
import os.path

from torch.utils.data import Dataset
import torch
from itertools import cycle
from torchvision import datasets, transforms
from PIL import Image, ImageStat
from constants.path_constants import *


class CustomImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder class. Workaround to swap class index assignment.
    """

    def __init__(self, dataset, transform=None, class_values=None):
        """
        :param dataset: (str) Dataset path
        :param transform: (torch.transforms) Set of transforms to be applied to input data
        :param class_values: (dict) definition of classes and numeric value used by the model
        """
        super(CustomImageFolder, self).__init__(dataset, transform=transform)
        if class_values:
            self.class_to_idx = class_values

    def __getitem__(self, index):
        """
        Extended method of ImageFolder
        :param index: (int) image index
        :return: Image, label and data info
        """
        sample, label = super(datasets.ImageFolder, self).__getitem__(index)
        return sample, label, index


def load_and_transform_base_data(stage, batch_size=1, shuffle=False):
    """
    Loads a dataset and applies the corresponding transformations
    :param stage: (str) Dataset to be loaded based on stage: train, test, validation (if any)
    :param batch_size: (int) number of batch for CAC and DR datasets
    :param shuffle: (bool) shuffle samples order within datasets
    """
    assert stage in ['train', 'test']
    # TODO: apply custom normalization

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loading CAC dataset and generate dataloader
    cac_dataset_path = os.path.join(os.path.abspath(CAC_DATASET_FOLDER), stage)
    cac_dataset = CustomImageFolder(cac_dataset_path,
                                    class_values={'CACSmenos400': 0, 'CACSmas400': 1},
                                    transform=data_transforms)
    cac_data_loader = torch.utils.data.DataLoader(cac_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=4)
    print(f'Loaded {len(cac_dataset)} images under {cac_dataset_path}: Classes: {cac_dataset.class_to_idx}')

    return cac_data_loader



