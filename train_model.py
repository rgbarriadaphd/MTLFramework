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
from dataset.base_dataset import load_and_transform_base_data
from dataset.mtl_dataset import load_and_transform_mtl_data
from utils.cnn import DLModel, train_mtl_model, evaluate_mtl_model, train_base_model, evaluate_base_model


class TrainMTLModel:

    def __init__(self, date_time):
        """
        Model train constructor initializes al class attributes.
        :param date_time: (str) date and time to identify execution
        """
        self._date_time = date_time
        self._mtl_model = None
        self._base_model = None

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
        self._mtl_model = DLModel(device=self._device, n_classes=7, load_model=LOAD_PREVIOUS_MODEL).get()
        self._mtl_model.to(device=self._device)
        if COMPARE_BASE_MODEL:
            self._base_model = DLModel(device=self._device, n_classes=2).get()
            self._base_model.to(device=self._device)

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

    def _save_losses_plot(self, cac_loss, dr_loss, base_loss):
        """
        Plots fold loss evolution
        :param cac_loss: (list) list of cac losses
        :param dr_loss: (list) list of dr losses
        """
        assert len(cac_loss) == len(dr_loss)

        if base_loss:
            assert len(cac_loss) == len(base_loss)
            plt.plot(list(range(1, len(base_loss) + 1)), base_loss, '-g', label='CAC (std alone')

        plt.plot(list(range(1, len(cac_loss) + 1)), cac_loss, '-b', label='CAC (MTL)')
        plt.plot(list(range(1, len(dr_loss) + 1)), dr_loss, '-r', label='DR (MTL)')

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='upper left')
        plt.title('Loss evolution')

        plt.legend(loc='upper right')

        plot_path = os.path.join(self._train_folder, 'loss.png')
        logging.info(f'Saving plot to {plot_path}')
        plt.savefig(plot_path)

    def _save_accuracies(self, control_accuracies, base_control_accuracies):
        """
        Plot together train and test accuracies by fold
        :param control_accuracies: (tuple) lists of accuracies: test and train for global, CAC and DR
        """
        train_cac, train_dr = control_accuracies[0], control_accuracies[1]
        test_cac, test_dr = control_accuracies[2], control_accuracies[3]
        assert len(train_cac) == len(train_dr) == len(test_cac) == len(test_dr)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        plt.title(f'Model accuracy')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        x_epochs = list(range(1, len(train_cac) + 1))
        ax1.plot(x_epochs, train_cac, label='CAC train (MTL)')
        # ax1.plot(x_epochs, train_dr, label='DR train (MTL)')

        ax1.plot(x_epochs, test_cac, label='CAC test (MTL)')
        # ax1.plot(x_epochs, test_dr, label='DR test (MTL)')

        if base_control_accuracies:
            train_base, test_base = base_control_accuracies[0], base_control_accuracies[1]
            assert len(train_cac) == len(train_base) == len(test_base)
            ax1.plot(x_epochs, train_base, label='CAC train (std alone)')
            ax1.plot(x_epochs, test_base, label='CAC test (std alone)')

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

        base_cac_accuracy = None
        base_losses = None
        base_control_accuracies = None
        if COMPARE_BASE_MODEL:
            # Train base model.
            # <--------------------------------------------------------------------->
            # Generate train data
            base_cac_train_dataloader = load_and_transform_base_data(stage='train',
                                                                     batch_size=8,
                                                                     shuffle=True)
            self._base_model, base_losses, base_control_accuracies = train_base_model(model=self._base_model,
                                                                                      device=self._device,
                                                                                      train_loader=base_cac_train_dataloader
                                                                                      )

            # Test base model.
            # <--------------------------------------------------------------------->
            # Generate test data
            base_cac_test_dataloader = load_and_transform_base_data(stage='test',
                                                                    batch_size=1,
                                                                    shuffle=False)  # shuffle does not matter for test

            base_cac_accuracy = evaluate_base_model(model=self._base_model,
                                                    device=self._device,
                                                    test_loader=base_cac_test_dataloader)

        # Train MTL model.
        # <--------------------------------------------------------------------->
        # Generate train data
        cac_train_data_loader, dr_train_data_loader = load_and_transform_mtl_data(stage='train',
                                                                                  batch_size=[8, 32],
                                                                                  shuffle=True)
        self._mtl_model, losses, control_accuracies = train_mtl_model(model=self._mtl_model,
                                                                      device=self._device,
                                                                      cac_train_loader=cac_train_data_loader,
                                                                      dr_train_loader=dr_train_data_loader
                                                                      )

        print("-----------------------------------------------------------------------")
        print("                         TEST                                          ")
        print("-----------------------------------------------------------------------")
        # Test MTL model.
        # <--------------------------------------------------------------------->

        # Generate test data
        cac_test_data_loader, dr_test_data_loader = load_and_transform_mtl_data(stage='test',
                                                                                batch_size=[1, 1],
                                                                                shuffle=False)  # shuffle does not matter for test

        cac_accuracy, dr_accuracy = evaluate_mtl_model(model=self._mtl_model,
                                                       device=self._device,
                                                       cac_test_loader=cac_test_data_loader,
                                                       dr_test_loader=dr_test_data_loader)

        print(f'[FINAL] CAC acc.: {cac_accuracy}')
        print(f'[FINAL] DR acc.: acc.: {dr_accuracy}')
        # Generate Loss plot
        if SAVE_LOSS_PLOT:
            self._save_losses_plot(losses[0], losses[1], base_losses)

        # Generate Loss plot
        if SAVE_ACCURACY_PLOT:
            self._save_accuracies(control_accuracies, base_control_accuracies)

        summary_data = {
            'execution_time': str(timedelta(seconds=time.time() - t0)),
            'cac_accuracy': cac_accuracy,
            'dr_accuracy': dr_accuracy,
            'cac_n_train': len(cac_train_data_loader.dataset),
            'cac_n_test': len(cac_test_data_loader.dataset),
            'dr_n_train': len(dr_train_data_loader.dataset),
            'dr_n_test': len(dr_test_data_loader.dataset),
            'base_accuracy': base_cac_accuracy
        }

        self._save_train_summary(summary_data)
