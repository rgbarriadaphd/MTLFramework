"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: train_model.py

Description: Class to handle train stages
"""
import time
from datetime import timedelta, datetime
import csv
from string import Template

import torch
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from constants.train_constants import *
from constants.path_constants import *
from dataset.base_dataset import load_and_transform_base_data
from dataset.mtl_dataset import load_and_transform_data
from utils.cnn import DLModel, train_model, evaluate_model


class TrainMTLModel:

    def __init__(self, date_time):
        """
        Model train constructor initializes al class attributes.
        :param date_time: (str) date and time to identify execution
        """
        self._date_time = date_time
        self._model = None

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
        self._model = DLModel(device=self._device).get()
        self._model.to(device=self._device)

    def _save_train_summary(self, summary_data):
        """
        Writes performance summnary
        :param summary_data: (dict) Contains model performance data
        """

        # Global Configuration
        summary_template_values = {
            'datetime': datetime.now(),
            'model': "MTL",
            'normalized': CUSTOM_NORMALIZED,
            'save_model': SAVE_MODEL,
            'plot_loss': SAVE_LOSS_PLOT,
            'epochs': EPOCHS,
            'batch_size': [(dt, DATASETS[dt]['batch_size']) for dt, values in DATASETS.items()],
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

    def _save_csv_data(self, train_data):
        """
        :param train_data: (tuple) lists of train data:
            train_data[0]: train loss
            train_data[1]: train accuracy over train dataset
            train_data[2]: train accuracy over test dataset
        """
        for i, csv_name in enumerate(['loss', 'accuracy_on_train', 'accuracy_on_test']):
            data = train_data[i]
            csv_path = os.path.join(self._train_folder, f'{csv_name}.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows([data])

    def run(self):
        """
        Run train stage
        """
        t0 = time.time()

        # Init model
        self._init_model()

        # Train MTL model.
        # <--------------------------------------------------------------------->
        # Generate train data
        train_data_loaders = load_and_transform_data(stage='train',
                                                     shuffle=True)
        self._model, train_data = train_model(model=self._model,
                                              device=self._device,
                                              train_loaders=train_data_loaders
                                              )



        self._save_csv_data(train_data)

        # Test MTL model.
        # <--------------------------------------------------------------------->
        # Generate test data
        test_data_loaders = load_and_transform_data(stage='test',
                                                    shuffle=False)  # shuffle does not matter for test
        accuracy = evaluate_model(model=self._model,
                                  device=self._device,
                                  test_loaders=test_data_loaders)


        summary_data = {
            'execution_time': str(timedelta(seconds=time.time() - t0)),
            'cac_accuracy': accuracy,
            'n_train': [(train_data_loaders[i].dataset.dataset_name, len(train_data_loaders[i].dataset)) for i in range(len(train_data_loaders))],
            'n_test': [(test_data_loaders[i].dataset.dataset_name, len(test_data_loaders[i].dataset)) for i in range(len(test_data_loaders))],
        }



        self._save_train_summary(summary_data)

        l_param = ["python plot/loss_plot.py", '"Loss evolution"', '"epochs, loss"', '"CAC loss"',
                   f'{os.path.join(self._train_folder, "loss.csv")}',
                   f'{os.path.join(self._train_folder, "loss.png")}']

        call = ' '.join(l_param)
        os.system(call)

        if CONTROL_TRAIN:
            l_param = ["python plot/loss_plot.py", '"Train progress"', '"epochs, accuracy"', '"train, test"',
                       f'{os.path.join(self._train_folder, "accuracy_on_train.csv")},{os.path.join(self._train_folder, "accuracy_on_test.csv")}',
                       f'{os.path.join(self._train_folder, "accuracy.png")}']
            os.system(' '.join(l_param))
