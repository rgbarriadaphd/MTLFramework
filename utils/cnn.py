"""
# Author: ruben 
# Date: 30/5/22
# Project: MTLFramework
# File: cnn.py

Description: Functions to deal with the cnn operations
"""
import logging

import torch
from torch import nn
from torchvision import models

from constants.train_constants import *

from torch.utils.tensorboard import SummaryWriter

def main():
    print("run")


class MultiTasksModel:
    """
    Class to manage the architecture initialization
    """

    def __init__(self, device, path=None):
        """
        Architecture class constructor
        :param path: (torch.device) Running device
        :param path: (str) If defined, then the model has to be loaded.
        """
        self._model_path = path
        self._device = device
        self._model = None

        if MODEL_SEED > 0:
            torch.manual_seed(MODEL_SEED)

        self._model = models.vgg16(pretrained=True)

        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, 7)
        features.extend([linear])
        self._model.classifier = nn.Sequential(*features)

        for param in self._model.parameters():
            param.requires_grad = REQUIRES_GRAD

        if self._model_path:
            self._model.load_state_dict(torch.load(self._model_path, map_location=torch.device(self._device)))
            logging.info(f'Loading architecture from {self._model_path}')

        print(self._model)

    def get(self):
        """
        Return model
        """
        return self._model


if __name__ == '__main__':
    main()
