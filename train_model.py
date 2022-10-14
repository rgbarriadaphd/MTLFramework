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
from PIL import Image
import hashlib

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
        self._dataset_history = {}
        self._model_history = {}
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
        if len(DATASETS) == 1:
            dataset_name = list(DATASETS.keys())[0]

        # Generate custom mean and std normalization values from only train dataset
        if not CUSTOM_NORMALIZED or not os.path.exists(DATASETS[dataset_name]['normalization_path']):
            self._normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return

        # Retrieve normalization file
        with open(DATASETS[dataset_name]['normalization_path']) as f:
            normalization_data = json.load(f)
        print("Custom Normalization!")
        self._normalization = (normalization_data[str(outer_fold_id)][str(inner_fold_id)]['mean'],
                               normalization_data[str(outer_fold_id)][str(inner_fold_id)]['std'])

    def _init_model(self):
        """
        Gathers model architecture
        :return:
        """
        self._model_obj = DLModel(device=self._device)
        self._model = self._model_obj.get()
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
            'model': [dt for dt in DATASETS],
            'normalized': CUSTOM_NORMALIZED,
            'save_model': SAVE_MODEL,
            'plot_loss': SAVE_LOSS_PLOT,
            'save_accuracy': SAVE_ACCURACY_PLOT,
            'control_train': CONTROL_TRAIN,
            'plot_roc': SAVE_ROC_PLOT,
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
            'outer_max': N_INCREASED_FOLDS[1],
            'dynamic_freeze': DYNAMIC_FREEZE,
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

    def _save_roc_data(self, outer_fold_id, inner_fold_id, fpr, tpr, roc_auc):
        """
        :param outer_fold_id: (int) outer fold identifier
        :param inner_fold_id: (int) inner fold identifier
        :param fpr: (dict) false positives ratio dictionary
        :param tpr: (dict) true positives ratio dictionary
        :param roc_auc: (dict) roc auc data
        """
        roc_data = {'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': roc_auc}
        json_file = os.path.join(self._csv_folder, f'roc_{outer_fold_id}_{inner_fold_id}.json')

        with open(json_file, 'w') as f:
            json.dump(roc_data, f)

    def _plot_loss_curves(self, outer_fold_id, inner_fold_id):
        """
        Plot losses
        """
        if not SAVE_LOSS_PLOT:
            return
        l_param = ["python plot/loss_plot.py", f'"Loss evolution (fold {outer_fold_id}-{inner_fold_id})"',
                   '"epochs, loss"', '"CAC loss"',
                   f'{os.path.join(self._csv_folder, "loss_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")}',
                   f'{os.path.join(self._plot_folder, "loss_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']

        call = ' '.join(l_param)
        os.system(call)

        if CONTROL_TRAIN:
            l_param = ["python plot/loss_plot.py", f'"Train progress(fold {inner_fold_id})"',
                       '"epochs, accuracy"', '"train, test"',
                       f'{os.path.join(self._csv_folder, "accuracy_on_train_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")},{os.path.join(self._csv_folder, "accuracy_on_test_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")}',
                       f'{os.path.join(self._plot_folder, "accuracy_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']
            os.system(' '.join(l_param))

    def _plot_roc_curves(self, outer_fold_id, inner_fold_id):
        """
        Plot losses
        """
        if not SAVE_ROC_PLOT:
            return
        json_file = os.path.join(self._csv_folder, f'roc_{outer_fold_id}_{inner_fold_id}.json')
        l_param = ["python plot/roc_plot.py", f'"ROC (fold {outer_fold_id}-{inner_fold_id})"',
                   '"False Positive Rate, True Positive Rate"', '"ROC curve (area = {{}})"',
                   json_file,
                   f'{os.path.join(self._plot_folder, "roc_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']
        call = ' '.join(l_param)
        os.system(call)

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

    def _model_health_checks(self):
        """
        Validates the mdoel evolution on each iteration
        """
        pres = []
        post = []
        for i in range(N_INCREASED_FOLDS[0], N_INCREASED_FOLDS[1] + 1):
            for j in range(1, 6):
                pres.append(self._model_history[(i, j)]['pre-train'])
                post.append(self._model_history[(i, j)]['post-train'])

        # All models starts
        print(self._model_history)
        assert len(list(set(pres))) == 1
        n_outer = N_INCREASED_FOLDS[1] + 1 - N_INCREASED_FOLDS[0]
        assert (len(list(set(post))) == n_outer * 5) and (pres[0] != post[0])

    def _folds_health_checks(self):
        """
        Validates the transition of test/train data over folds
        """
        n_outer = N_INCREASED_FOLDS[1] + 1 - N_INCREASED_FOLDS[0]
        print(f'n_outer ---> {n_outer}')
        expected_train_appearances = n_outer * 4
        expected_test_appearances = n_outer * 1
        assert (expected_train_appearances + expected_test_appearances) == 5 * n_outer
        for image_data in self._dataset_history:
            image_name = image_data
            image_labels = self._dataset_history[image_data]['label']
            uses_train = self._dataset_history[image_data]['train']
            uses_test = self._dataset_history[image_data]['test']
            # Check completeness and coherence
            sum_cases = uses_train + uses_test
            assert len(sum_cases) == 5 * n_outer
            for i in range(N_INCREASED_FOLDS[0], N_INCREASED_FOLDS[1] + 1):
                for j in range(1, 6):
                    assert (i, j) in sum_cases
            # check lengths
            assert len(image_labels) == len(uses_train) + len(uses_test), f'{len(image_labels)}| {len(uses_train)}|{len(uses_test)}'
            assert len(uses_train) + len(uses_test) == 5 * n_outer, f'{len(uses_train)}| {len(uses_test)}|{n_outer}'
            # only one value 1/0 acceptable as label
            assert len(set(image_labels)) == 1, f'{len(set(image_labels))}'
            # expected appearances of the image in the train set
            assert len(set(uses_train)) == expected_train_appearances, f'{len(set(uses_train))}| {expected_train_appearances}'
            # expected appearances of the image in the test set
            assert len(set(uses_test)) == expected_test_appearances, f'{len(set(uses_test))}| {expected_test_appearances}'

    def _check_hash_dynamic_run(self):
        train_images_list = []
        train_hash_list = []
        for root, dirs, files in os.walk(os.path.join(DYNAMIC_RUN_FOLDER, 'train')):
            for image_name in files:
                assert os.path.exists(os.path.join(root, image_name))
                image_path = os.path.join(root, image_name)
                pil_image = Image.open(image_path)
                md5hash = hashlib.md5(pil_image.tobytes())
                train_images_list.append(image_path)
                train_hash_list.append(md5hash.hexdigest())
        assert len(train_images_list) == len(train_hash_list)
        assert len(train_hash_list) == len(set(train_hash_list))

        test_images_list = []
        test_hash_list = []
        for root, dirs, files in os.walk(os.path.join(DYNAMIC_RUN_FOLDER, 'test')):
            for image_name in files:
                assert os.path.exists(os.path.join(root, image_name))
                image_path = os.path.join(root, image_name)
                pil_image = Image.open(image_path)
                md5hash = hashlib.md5(pil_image.tobytes())
                test_images_list.append(image_path)
                test_hash_list.append(md5hash.hexdigest())
        assert len(test_images_list) == len(test_hash_list)
        assert len(test_hash_list) == len(set(test_hash_list))

        intersection = list(set(train_hash_list) & set(test_hash_list))
        assert len(intersection) == 0

        print(f'Dynamic Run folder OK!! --> {DYNAMIC_RUN_FOLDER}')

    def _update_dataset_history(self, train_data, test_data, outer_fold_id, inner_fold_id):
        """
        Updates dataset history
        :param train_data:
        :param test_data:
        :param outer_fold_id:
        :param inner_fold_id:
        :return:
        """
        for label in train_data:
            for image in train_data[label]:
                if image not in self._dataset_history:
                    self._dataset_history[image] = {}
                    self._dataset_history[image]['train'] = [(outer_fold_id, inner_fold_id)]
                    self._dataset_history[image]['test'] = []
                    self._dataset_history[image]['label'] = [0 if label == 'CEN' else 1]
                    continue
                self._dataset_history[image]['train'].append((outer_fold_id, inner_fold_id))
                self._dataset_history[image]['label'].append(0 if label == 'CEN' else 1)
        for label in test_data:
            for image in test_data[label]:
                if image not in self._dataset_history:
                    self._dataset_history[image] = {}
                    self._dataset_history[image]['train'] = []
                    self._dataset_history[image]['test'] = [(outer_fold_id, inner_fold_id)]
                    self._dataset_history[image]['label'] = [0 if label == 'CEN' else 1]
                    continue
                self._dataset_history[image]['test'].append((outer_fold_id, inner_fold_id))
                self._dataset_history[image]['label'].append(0 if label == 'CEN' else 1)

    def _compute_cross_validation_performance(self):
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
                fold_train_data, fold_test_data = self._fold_handler.generate_run_set(inner_fold_id)
                print(f'Generate run set: {time.time() - t0}s')
                self._check_hash_dynamic_run()
                self._update_dataset_history(fold_train_data, fold_test_data, outer_fold_id, inner_fold_id)

                # Init model
                t0 = time.time()
                self._init_model()
                print(f'Init the model: {time.time() - t0}s')
                self._model_history[(outer_fold_id, inner_fold_id)] = {'pre-train': self._model_obj.get_control_model(),
                                                                       'post-train': None}

                # Get dataset normalization mean and std
                t0 = time.time()
                self._get_normalization(outer_fold_id, inner_fold_id)
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
                                                      mean=self._normalization[0],
                                                      std=self._normalization[1],
                                                      train_loaders=train_data_loaders
                                                      )
                tf_fold_train = time.time() - t0_fold_train
                self._model_history[(outer_fold_id, inner_fold_id)]['post-train'] = self._model_obj.get_control_model()
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
                print(f'Fold accuracy on test dataset: {fold_accuracy}')
                self._folds_acc.append(fold_accuracy)

                self._save_roc_data(outer_fold_id, inner_fold_id,
                                    fold_performance[f'fpr_{outer_fold_id}_{inner_fold_id}'],
                                    fold_performance[f'tpr_{outer_fold_id}_{inner_fold_id}'],
                                    fold_performance[f'roc_auc_{outer_fold_id}_{inner_fold_id}']
                                    )

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

                # Plot ROC curves
                self._plot_roc_curves(outer_fold_id, inner_fold_id)

                # Run only one fold
                if MONO_FOLD:
                    logging.info("Only one fold is executed")
                    break
            if MONO_FOLD:
                break

        # Check model consistency
        self._folds_health_checks()
        self._model_health_checks()

        # Compute global performance
        self._global_performance = self._compute_cross_validation_performance()

        # Output final summary
        self._save_train_summary()
