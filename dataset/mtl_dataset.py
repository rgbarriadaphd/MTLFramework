"""
# Author: ruben
# Date: 23/5/22
# Project: MTLFramework
# File: mtl_dataset.py

Description: Class that implements Multi-Task Learning dataset
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


def load_and_transform_data(stage, batch_size=[1, 1], shuffle=False):
    """
    Loads a dataset and applies the corresponding transformations
    :param stage: (str) Dataset to be loaded based on stage: train, test, validation (if any)
    :param batch_size: (list of int) number of batch for CAC and DR datasets
    :param shuffle: (list of int) shuffle samples order within datasets
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
                                                  batch_size=batch_size[0],
                                                  shuffle=shuffle,
                                                  num_workers=4)
    print(f'Loaded {len(cac_dataset)} images under {cac_dataset_path}: Classes: {cac_dataset.class_to_idx}')

    # Loading DR dataset and generate dataloader

    dr_dataset_path = os.path.join(os.path.abspath(DR_DATASET_FOLDER), stage)
    dr_dataset = CustomImageFolder(dr_dataset_path,
                                   transform=data_transforms)

    dr_data_loader = torch.utils.data.DataLoader(dr_dataset,
                                                 batch_size=batch_size[1],
                                                 shuffle=shuffle,
                                                 num_workers=4)

    print(f'Loaded {len(dr_dataset)} images under {dr_dataset_path}: Classes: {dr_dataset.class_to_idx}')

    return cac_data_loader, dr_data_loader


if __name__ == '__main__':

    cac_data_loader, dr_data_loader = load_and_transform_data(stage='train',
                                                              batch_size=[8, 32],
                                                              shuffle=True)

    epochs = 2
    for i in range(epochs):
        print(f'Epoch = {i}')
        for i, (data1, data2) in enumerate(zip(cycle(cac_data_loader), dr_data_loader)):
            image1 = data1[0]
            label1 = data1[1]
            index1 = data1[2]

            image2 = data2[0]
            label2 = data2[1]
            index2 = data2[2]

            print(f'CAC samples: {index1} | Labels: {label1}')
            print(f'DR samples: {index2} | Labels: {label2}')
            print("...")

            # First train CAC batch + backprop with common loss
            # First train DR batch + backprop with common loss
