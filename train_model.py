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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from constants.train_constants import *
from constants.path_constants import *
from dataset.mtl_dataset import load_and_transform_data
from utils.cnn import MultiTasksModel, train_model, evaluate_model


class TrainModel:

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

    def _save_losses_plot(self, general_loss, cac_loss, dr_loss):
        """
        Plots fold loss evolution
        :param general_loss: (list) list of general losses
        :param cac_loss: (list) list of cac losses
        :param dr_loss: (list) list of dr losses
        """
        assert len(general_loss) == len(cac_loss) == len(dr_loss)

        plt.plot(list(range(1, len(general_loss) + 1)), general_loss, '-g', label='Global')
        plt.plot(list(range(1, len(cac_loss) + 1)), cac_loss, '-b', label='CAC')
        plt.plot(list(range(1, len(dr_loss) + 1)), dr_loss, '-r', label='DR')

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='upper left')
        plt.title('Loss evolution')

        plt.legend(loc='upper right')

        plot_path = os.path.join(self._train_folder, 'loss.png')
        logging.info(f'Saving plot to {plot_path}')
        plt.savefig(plot_path)

    def _save_fold_accuracies(self, control_accuracies):
        """
        Plot together train and test accuracies by fold
        :param control_accuracies: (tuple) lists of accuracies: test and train for global, CAC and DR
        """
        train_global, train_cac, train_dr = control_accuracies[0], control_accuracies[1], control_accuracies[2]
        test_global, test_cac, test_dr = control_accuracies[3], control_accuracies[4], control_accuracies[5]
        assert len(train_global) == len(test_global) == len(train_cac) == len(train_dr) == len(test_cac) == len(test_dr)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        plt.title(f'Model accuracy')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        x_epochs = list(range(1, len(train_global) + 1))
        ax1.plot(x_epochs, train_global, label='Global train')
        ax1.plot(x_epochs, train_cac, label='CAC train')
        ax1.plot(x_epochs, train_dr, label='DR train')

        ax1.plot(x_epochs, test_global, label='Global test')
        ax1.plot(x_epochs, test_cac, label='CAC test')
        ax1.plot(x_epochs, test_dr, label='DR test')

        ax1.legend()

        plot_path = os.path.join(self._train_folder, f'control_accuracy.png')
        logging.info(f'Saving accuracy plot to {plot_path}')
        plt.savefig(plot_path)

    def run(self):
        """
        Run train stage
        """
        t0 = time.time()

        # Init model
        self._init_model()
        self._model.to(device=self._device)

        # Train model.
        # <--------------------------------------------------------------------->
        # Generate train data
        cac_train_data_loader, dr_train_data_loader = load_and_transform_data(stage='train',
                                                                              batch_size=[8, 32],
                                                                              shuffle=True)
        self._model, losses, control_accuracies = train_model(model=self._model,
                                                              device=self._device,
                                                              cac_train_loader=cac_train_data_loader,
                                                              dr_train_loader=dr_train_data_loader
                                                              )

        # Test model.
        # <--------------------------------------------------------------------->

        # Generate test data
        cac_test_data_loader, dr_test_data_loader = load_and_transform_data(stage='test',
                                                                            batch_size=[1, 1],
                                                                            shuffle=False)  # shuffle does not matter for test

        global_accuracy, cac_accuracy, dr_accuracy = evaluate_model(model=self._model,
                                                                    device=self._device,
                                                                    cac_test_loader=cac_test_data_loader,
                                                                    dr_test_loader=dr_test_data_loader)

        # Generate Loss plot
        if SAVE_LOSS_PLOT:
            self._save_losses_plot(losses[0], losses[1], losses[2])

        # Generate Loss plot
        if SAVE_ACCURACY_PLOT:
            self._save_fold_accuracies(control_accuracies)

        summary_data = {
            'execution_time': str(timedelta(seconds=time.time() - t0)),
            'model_accuracy': global_accuracy,
            'cac_accuracy': cac_accuracy,
            'dr_accuracy': dr_accuracy,
            'cac_n_train': len(cac_train_data_loader),
            'cac_n_test': len(cac_test_data_loader),
            'dr_n_train': len(dr_train_data_loader),
            'dr_n_test': len(dr_test_data_loader),
        }

        self._save_train_summary(summary_data)
