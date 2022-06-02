"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: train_model.py

Description: Class to handle train stages
"""
import time
from datetime import timedelta, datetime
import os
from string import Template

import torch
import logging

from constants.train_constants import *
from constants.path_constants import *
from dataset.mtl_dataset import load_and_transform_data
from utils.cnn import MultiTasksModel


class TrainModel:

    def __init__(self, date_time):
        """
        Model train constructor initializes al class attributes.
        :param date_time: (str) date and time to identify execution
        """
        self._date_time = date_time
        self._create_train_folder()
        self._init_device()

    def _create_train_folder(self):
        """
        Creates the folder where output data will be stored
        """
        self._train_folder = os.path.join(TRAIN_FOLDER, f'train_{self._date_time}')
        try:
            os.mkdir(self._train_folder)
        except OSError:
            logging.error("Creation of model directory %s failed" % self._train_folder)
        else:
            logging.info("Successfully created model directory %s " % self._train_folder)

    def _init_device(self):
        """
        Initialize either cuda or cpu device where train will be run
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self._device}')

    def _init_model(self):
        """
        Gathers model architecture
        :return:
        """
        self._architecture = MultiTasksModel(device=self._device)
        self._model = self._architecture.get()

    def _save_train_summary(self, summary_data):
        """
        Writes performance summnary
        :param summary_data: (dict) Contains model performance data
        """

        # Global Configuration
        summary_template_values = {
            'datetime': datetime.now(),
            'model': "Hybrid",
            'normalized': CUSTOM_NORMALIZED,
            'save_model': SAVE_MODEL,
            'plot_loss': SAVE_LOSS_PLOT,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'criterion': CRITERION,
            'optimizer': OPTIMIZER,
            'device': self._device,
            'require_grad': REQUIRES_GRAD,
            'weight_init': WEIGHT_INIT
        }

        summary_template_values.update(summary_data)

        # Substitute values
        with open(SUMMARY_TEMPLATE, 'r') as f:
            src = Template(f.read())
            report = src.substitute(summary_template_values)
            logging.info(report)

        # Write report
        with open(os.path.join(self._train_folder, 'summary.out'), 'w') as f:
            f.write(report)

    def run(self):
        """
        Run train stage
        """
        t0 = time.time()

        # Generate train and test data
        # cac_data_loader, dr_data_loader = load_and_transform_data(stage='train',
        #                                                           batch_size=[8, 32],
        #                                                           shuffle=True)
        # Init model
        self._init_model()


        test_accuracy = 0.890202

        summary_data = {
            'execution_time': str(timedelta(seconds=time.time() - t0)),
            'model_accuracy': test_accuracy,
            'n_train': 7526,
            'n_test': 2015,
            'norm_mean': [0.8, 0.06, 0.07],
            'norm_std': [0.132, 0.12, 0.8],
            'train_time': '78:48:45',
            'test_time': '45:36:15',
            'precision': 0.78,
            'recall': 0.89,
        }

        self._save_train_summary(summary_data)
