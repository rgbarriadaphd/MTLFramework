"""
# Author: ruben
# Date: 23/5/22
# Project: MTLFramework
# File: mtl_dataset.py

Description: Class that implements Multi-Task Learning dataset
"""
import random

from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageStat
import numpy as np
from constants.path_constants import *
from constants.train_constants import *


class CustomImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder class. Workaround to swap class index assignment.
    """

    def __init__(self, dataset, dataset_name, transform=None, class_values=None):
        """
        :param dataset: (str) Dataset path
        :param dataset_name: (str) identifier name of the dataset
        :param transform: (torch.transforms) Set of transforms to be applied to input data
        :param class_values: (dict) definition of classes and numeric value used by the model
        """
        self.class_values = class_values
        super(CustomImageFolder, self).__init__(dataset, transform=transform)
        self.dataset_name = dataset_name

    def find_classes(self, root):
        if self.class_values:
            self.class_to_idx = self.class_values

        return self.class_to_idx.keys(), self.class_to_idx

    def __getitem__(self, index):
        """
        Extended method of ImageFolder
        :param index: (int) image index
        :return: Image, label and data info
        """
        sample, label = super(datasets.ImageFolder, self).__getitem__(index)
        return sample, label, index, self.dataset_name


def load_and_transform_data(stage, shuffle=False, mean=None, std=None):
    """
    Loads a dataset and applies the corresponding transformations
    :param stage: (str) Dataset to be loaded based on stage: train, test, validation (if any)
    :param shuffle: (list of int) shuffle samples order within datasets
    :param mean: (list) Normalized mean
    :param std: (list) Normalized std
    """
    assert stage in ['train', 'test']

    if mean is None and std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        print(f'Applying custom normalization: mean={mean}, std={std}')

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset_loaders = []
    for dataset_name, data in DATASETS.items():
        dataset_path = os.path.join(os.path.abspath(DATASETS[dataset_name]['path']), stage)
        dataset = CustomImageFolder(dataset_path,
                                    dataset_name,
                                    class_values=DATASETS[dataset_name]['class_values'],
                                    transform=data_transforms)

        bs = DATASETS[dataset_name]['batch_size'] if stage == 'train' else 1
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=bs,
                                                  shuffle=shuffle,
                                                  num_workers=4)
        print(f'Loaded {len(dataset)} images under {dataset_path}: Classes: {dataset.class_to_idx}')
        dataset_loaders.append(data_loader)

    return dataset_loaders


def concat_datasets(batch_dataset_1, batch_dataset_2):
    # Concatenate both datasets
    concat_image = torch.cat((batch_dataset_1[0], batch_dataset_2[0]), 0)
    concat_label = torch.cat((batch_dataset_1[1], batch_dataset_2[1]), 0)
    concat_index = torch.cat((batch_dataset_1[2], batch_dataset_2[2]), 0)
    concat_dt_name = batch_dataset_1[3] + batch_dataset_2[3]

    before_shuffle_index = [concat_index[elem] for elem in range(len(concat_index))]

    # Shuffle data
    selection = list(range(len(concat_dt_name)))
    random.shuffle(selection)

    concat_image = concat_image[selection]
    concat_label = concat_label[selection]
    concat_index = concat_index[selection]
    concat_dt_name = list(concat_dt_name)
    concat_dt_name = [concat_dt_name[elem] for elem in selection]

    # Check shuffle
    after_shuffle_index = [concat_index[selection.index(elem)] for elem in range(len(concat_index))]
    assert before_shuffle_index == after_shuffle_index

    return concat_image, concat_label, concat_index, concat_dt_name


if __name__ == '__main__':
    data_loaders = load_and_transform_data(stage='train', shuffle=True)
