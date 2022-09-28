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
import json
import torch
import logging

from constants.train_constants import *
from constants.path_constants import *
from dataset.mtl_dataset import load_and_transform_data
from utils.cnn import DLModel, train_model, evaluate_model
from utils.fold_handler import FoldHandler
from utils.metrics import CrossValidationMeasures


class TrainMTLModel:

    def __init__(self, date_time):
        """
        Model train constructor initializes al class attributes.
        :param date_time: (str) date and time to identify execution
        """
        self._date_time = date_time
        self._model = None
        self._t0 = None
        self._folds_performance = []
        self._folds_acc = []
        self._global_performance = {}

        self._create_train_folder()
        self._generate_summary_template()
        self._init_device()

    def _create_train_folder(self):
        """
        Creates the folder where output data will be stored
        """
        self._train_folder = os.path.join(TRAIN_FOLDER, f'train_{self._date_time}')
        self._plot_folder = os.path.join(self._train_folder, PLOT_FOLDER)
        self._csv_folder = os.path.join(self._train_folder, CSV_FOLDER)
        self._fold_folder = os.path.join(self._train_folder, FOLD_FOLDER)
        try:
            os.mkdir(self._train_folder)
            os.mkdir(self._plot_folder)
            os.mkdir(self._csv_folder)
            os.mkdir(self._fold_folder)
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

    def _generate_fold_data(self, outer_fold_id):
        """
        Creates fold structure where train images will be split in 4 train folds + 1 test fold
        :param outer_fold_id: (int) Outer fold identifier
        """
        self._fold_dataset = os.path.join(ROOT_ORIGINAL_FOLDS, 'outer_fold_{}'.format(outer_fold_id))
        self._fold_handler = FoldHandler(self._fold_dataset, DYNAMIC_RUN_FOLDER)

    def _get_normalization(self, outer_fold_id, inner_fold_id):
        """
        Retrieves custom normalization defined in json. The established by Pytorch otherwise.
        :param outer_fold_id: (int) outer folder id
        :param inner_fold_id: (int) inner folder id
        :return: ((list) mean, (list) std) Normalized mean and std according to train dataset
        """
        # Generate custom mean and std normalization values from only train dataset
        if not CUSTOM_NORMALIZED or not os.path.exists(CAC_NORMALIZATION):
            self._normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return

        # Retrieve normalization file
        with open(CAC_NORMALIZATION) as f:
            normalization_data = json.load(f)
        self._normalization = (normalization_data[outer_fold_id][inner_fold_id]['mean'],
                               normalization_data[outer_fold_id][inner_fold_id]['std'])

    def _init_model(self):
        """
        Gathers model architecture
        :return:
        """
        self._model = DLModel(device=self._device).get()
        self._model.to(device=self._device)

    def _generate_summary_template(self):
        """
        Generates the summray template on the fly based on number of outer running folds
        """
        # Read base template
        # Substitute values
        tmp_lines = []
        min = N_INCREASED_FOLDS[0]
        max = N_INCREASED_FOLDS[1]
        with open(BASE_SUMMARY_TEMPLATE, 'r') as f:
            lines = f.readlines()

            for line in lines:
                if any([elem in line for elem in [f'_{i}_' for i in range(1, 11)]]):

                    for i in range(1, 11):
                        if f'_{i}_' in line:
                            n = i
                            break
                    if min <= n <= max:
                        tmp_lines.append(line)
                else:
                    tmp_lines.append(line)
        # Write tmpl
        with open(SUMMARY_TEMPLATE, 'w') as f:
            f.writelines(tmp_lines)

    def _save_train_summary(self, unique_fold_performance=None,
                            outer_fold=None, inner_fold=None):
        """
        Writes performance summary
        :param unique_fold_performance: (dict) Contains specific folder performance data
        :param outer_fold: (int) outer folder id
        :param inner_fold: (int) inner folder id
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
            'weight_init': WEIGHT_INIT,
            'outer_min': N_INCREASED_FOLDS[0],
            'outer_max': N_INCREASED_FOLDS[1]
        }

        if unique_fold_performance:
            aux_unique_fold_performance = {}
            for key, value in unique_fold_performance.items():
                # remove last indexes
                key_split = key.split('_')
                if key.startswith('fold_id_'):
                    aux_unique_fold_performance['outer_fold_id'] = key_split[len(key_split) - 2]
                    aux_unique_fold_performance['inner_fold_id'] = key_split[len(key_split) - 1]
                    continue

                aux_key = "_".join(key_split[:len(key_split) - 2])
                aux_unique_fold_performance[aux_key] = value
            tpl_file = SUMMARY_UNIQUE_FOLD_TEMPLATE
            out_path = os.path.join(self._fold_folder, f'summary_{outer_fold}_{inner_fold}.out')
            summary_template_values.update(aux_unique_fold_performance)
        else:
            tpl_file = SUMMARY_TEMPLATE.format(N_INCREASED_FOLDS[0])
            out_path = os.path.join(self._train_folder, 'summary.out')
            for fold in self._folds_performance:
                summary_template_values.update(fold)
            summary_template_values.update(self._global_performance)

        # Substitute values
        with open(tpl_file, 'r') as f:
            src = Template(f.read())
            report = src.substitute(summary_template_values)
            logging.info(report)

        # Write report
        with open(out_path, 'w') as f:
            f.write(report)

    def _save_csv_data(self, train_data, outer_fold_id, inner_fold_id):
        """
        :param train_data: (tuple) lists of train data:
            train_data[0]: train loss
            train_data[1]: train accuracy overtrain dataset
            train_data[2]: train accuracy over test dataset
        :param outer_fold_id: (int) outer fold identifier
        :param inner_fold_id: (int) inner fold identifier
        """
        for i, csv_name in enumerate(['loss', 'accuracy_on_train', 'accuracy_on_test']):
            data = train_data[i]
            csv_path = os.path.join(self._csv_folder, f'{csv_name}_{outer_fold_id}_{inner_fold_id}.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows([data])

    def _plot_loss_curves(self, outer_fold_id, inner_fold_id):
        """
        Plot losses
        """
        l_param = ["python plot/loss_plot.py", f'"Loss evolution (fold {outer_fold_id}-{inner_fold_id})"',
                   '"epochs, loss"', '"CAC loss"',
                   f'{os.path.join(self._csv_folder, "loss_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")}',
                   f'{os.path.join(self._plot_folder, "loss_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']

        call = ' '.join(l_param)
        os.system(call)

        if CONTROL_TRAIN:
            l_param = ["python plot/loss_plot.py", f'"Train progress(fold {inner_fold_id})"',
                       '"epochs, accuracy"', '"train, test"',
                       f'{os.path.join(self._csv_folder, "accuracy_on_train" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")},{os.path.join(self._csv_folder, "accuracy_on_test.csv")}',
                       f'{os.path.join(self._plot_folder, "accuracy" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']
            os.system(' '.join(l_param))

    def _get_fold_perform_data(self, outer_fold_id, inner_fold_id, train_data_loaders, test_data_loaders, tf_fold_train,
                               tf_fold_test):
        """
        Returns fold data in dict format
        """
        return {
            f'fold_id_{outer_fold_id}_{inner_fold_id}': inner_fold_id,
            f'n_train_{outer_fold_id}_{inner_fold_id}': [
                (train_data_loaders[i].dataset.dataset_name, len(train_data_loaders[i].dataset)) for i in
                range(len(train_data_loaders))],
            f'n_test_{outer_fold_id}_{inner_fold_id}': [
                (test_data_loaders[i].dataset.dataset_name, len(test_data_loaders[i].dataset)) for i in
                range(len(test_data_loaders))],
            f'mean_{outer_fold_id}_{inner_fold_id}': f'[{self._normalization[0][0]:.{ND}f}, {self._normalization[0][1]:.{ND}f}, {self._normalization[0][2]:.{ND}f}]',
            f'std_{outer_fold_id}_{inner_fold_id}': f'[{self._normalization[1][0]:.{ND}f}, {self._normalization[1][1]:.{ND}f}, {self._normalization[1][2]:.{ND}f}]',
            f'fold_train_time_{outer_fold_id}_{inner_fold_id}': f'{tf_fold_train:.{ND}f}',
            f'fold_test_time_{outer_fold_id}_{inner_fold_id}': f'{tf_fold_test:.{ND}f}',
        }

    def _compute_cross_validation_performance(self, folds_acc):
        """
        Computes cross validation metrics and returns in
        """
        # Compute global performance info
        cvm = CrossValidationMeasures(measures_list=self._folds_acc, percent=True, formatted=True)
        f_acc = '['
        for p, fa in enumerate(self._folds_acc):
            f_acc += f'{fa:.{ND}f}'
            if (p + 1) != len(self._folds_acc):
                f_acc += ','
        f_acc += ']'
        return {
            'folds_accuracy': f_acc,
            'cross_v_mean': cvm.mean(),
            'cross_v_stddev': cvm.stddev(),
            'cross_v_interval': cvm.interval(),
            'execution_time': str(timedelta(seconds=time.time() - self._t0))
        }

    def run(self):
        """
        Run train stage
        """
        self._t0 = time.time()

        for outer_fold_id in range(N_INCREASED_FOLDS[0], N_INCREASED_FOLDS[1] + 1):
            self._generate_fold_data(outer_fold_id)
            for inner_fold_id in range(1, 6):
                print(
                    f'***************************** Fold ID: {outer_fold_id} - {inner_fold_id} *****************************')
                # Generate fold data
                t0 = time.time()
                self._fold_handler.generate_run_set(inner_fold_id)
                print(f'Generate run set: {time.time() - t0}s')

                # Init model
                t0 = time.time()
                self._init_model()
                print(f'Init the model: {time.time() - t0}s')

                # Get dataset normalization mean and std
                t0 = time.time()
                self._get_normalization()
                print(f'Normalization: {time.time() - t0}s')

                # Train MTL model.
                # <--------------------------------------------------------------------->
                # Generate train data
                train_data_loaders = load_and_transform_data(stage='train',
                                                             shuffle=True,
                                                             mean=self._normalization[0],
                                                             std=self._normalization[1]
                                                             )
                t0_fold_train = time.time()
                self._model, train_data = train_model(model=self._model,
                                                      device=self._device,
                                                      train_loaders=train_data_loaders
                                                      )
                tf_fold_train = time.time() - t0_fold_train

                self._save_csv_data(train_data, outer_fold_id, inner_fold_id)

                # Test MTL model.
                # <--------------------------------------------------------------------->
                # Generate test data
                test_data_loaders = load_and_transform_data(stage='test',
                                                            mean=self._normalization[0],
                                                            std=self._normalization[1],
                                                            shuffle=False)  # shuffle does not matter for test
                t0_fold_test = time.time()
                fold_performance, fold_accuracy = evaluate_model(model=self._model,
                                                                 device=self._device,
                                                                 test_loaders=test_data_loaders,
                                                                 outer_fold_id=outer_fold_id,
                                                                 inner_fold_id=inner_fold_id)
                tf_fold_test = time.time() - t0_fold_test

                self._folds_acc.append(fold_accuracy)

                fold_data = self._get_fold_perform_data(outer_fold_id, inner_fold_id, train_data_loaders,
                                                        test_data_loaders, tf_fold_train,
                                                        tf_fold_test)
                # Update fold data
                fold_performance.update(fold_data)
                # Save partial fold data
                self._save_train_summary(unique_fold_performance=fold_performance,
                                         outer_fold=outer_fold_id,
                                         inner_fold=inner_fold_id)
                self._folds_performance.append(fold_performance)

                # Plot loss curves
                self._plot_loss_curves(outer_fold_id, inner_fold_id)

        # Compute global performance
        self._global_performance = self._compute_cross_validation_performance()

        # Output final summary
        self._save_train_summary()
