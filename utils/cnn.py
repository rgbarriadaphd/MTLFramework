"""
# Author: ruben 
# Date: 30/5/22
# Project: MTLFramework
# File: cnn.py

Description: Functions to deal with the cnn operations
"""
import logging
import math
from itertools import cycle
import sys
import random
import numpy as np

import torch
from torch import nn
from torchvision import models
from torch import optim
from tqdm import tqdm
from constants.train_constants import *
from dataset.mtl_dataset import load_and_transform_data
from utils.metrics import PerformanceMetrics


def main():
    print("run")


class DLModel:
    """
    Class to manage the architecture initialization
    """

    def __init__(self, device, path=None):
        """
        Architecture class constructor
        :param device: (torch.device) Running device
        """
        self._device = device
        self._model = None
        n_classes = sum([len(DATASETS[dt]['class_values']) for dt in DATASETS])

        if MODEL_SEED > 0:
            torch.manual_seed(MODEL_SEED)

        self._model = models.vgg16(pretrained=True)

        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, n_classes)
        features.extend([linear])
        self._model.classifier = nn.Sequential(*features)

        for param in self._model.parameters():
            param.requires_grad = REQUIRES_GRAD

    def get(self):
        """
        Return model
        """
        return self._model


class MTLRetinalSelectorLoss(nn.Module):
    def __init__(self):
        super(MTLRetinalSelectorLoss, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._n_classes = sum([len(DATASETS[dt]['class_values']) for dt in DATASETS])

    def _softmax(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x), axis=0)

    def _cross_entropy(self, y, y_pre, selector):
        loss = -torch.sum(selector * (y * torch.log(y_pre) + (1 - y) * torch.log(1 - y_pre)))
        return loss / float(y_pre.shape[0])

    def forward(self, predictions, grounds, dataset_names=None):
        assert len(predictions) == len(grounds) == len(dataset_names)
        loss = 0.0

        for i in range(len(predictions)):

            prediction = predictions[i]
            ground = grounds[i]
            data_type = dataset_names[i]

            output_probs = self._softmax(prediction)

            one_hot_ground = torch.zeros(self._n_classes)
            one_hot_ground[ground.item()] = 1.0
            one_hot_ground = one_hot_ground.to(device=self._device)

            selector = torch.tensor(DATASETS[data_type]['selector'])
            selector = selector.to(device=self._device)

            partial_loss = self._cross_entropy(one_hot_ground,
                                               output_probs,
                                               selector)
            if math.isnan(partial_loss):
                partial_loss = torch.tensor(0.0)
                partial_loss = partial_loss.to(device=self._device)

            # print("-----------------------------------")
            # print(f'Pred: {prediction}')
            # print(f'Softmax: {output_probs}')
            # print(f'Labels: {ground}')
            # print(f'Dataset: {data_type}')
            # print(f'One hot ground: {one_hot_ground}')
            # print(f'Selector: {selector}')
            # print(f'Partial loss: {partial_loss}')
            # print("-----------------------------------")
            loss += partial_loss

        # print(f'Total loss: {loss}')
        # print("-----------------------------------")
        return loss


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

    # Chek shuffle
    after_shuffle_index = [concat_index[selection.index(elem)] for elem in range(len(concat_index))]
    assert before_shuffle_index == after_shuffle_index

    return concat_image, concat_label, concat_index, concat_dt_name


def train_model(model, device, train_loaders):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param train_loaders: (list torchvision.datasets) List of  train dataloader containing dataset images
    :return: train model, losses array, accuracies of test and train datasets
    """
    n_train = sum([len(train_loaders[i].dataset) for i in range(len(train_loaders))])

    logging.info(f'''Starting training:
            Epochs:          {EPOCHS}
            Batch size:      {[(dt, DATASETS[dt]['batch_size']) for dt, values in DATASETS.items()]}
            Learning rate:   {LEARNING_RATE}
            Training sizes:   {[(train_loaders[i].dataset.dataset_name, len(train_loaders[i].dataset)) for i in range(len(train_loaders))]}
            Device:          {device.type}
            Criterion:       {CRITERION}
            Optimizer:       {OPTIMIZER}
        ''')

    losses = []
    test_accuracy_list = []
    train_accuracy_list = []

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if CRITERION == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif CRITERION == 'MTLRetinalSelectorLoss':
        criterion = MTLRetinalSelectorLoss()

    inf_condition = False
    for epoch in range(EPOCHS):
        # if inf_condition:
        #     logging.info(f'outter INF condition at epoch {epoch + 1}')
        #     break
        model.train(True)
        running_loss = 0.0
        print(f'------------- EPOCH: {epoch + 1} -------------')

        if len(DATASETS) > 1:
            # Assuming CAC=0, DR=1
            dataset_iterator = zip(cycle(train_loaders[0]), train_loaders[1])
        else:
            dataset_iterator = train_loaders[0]

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(dataset_iterator):
                # if inf_condition:
                #     logging.info(f'inner INF condition at epoch {epoch + 1}')
                #     break
                sample, ground, index, dt_name = concat_datasets(batch[0], batch[1]) if len(DATASETS) > 1 else batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                current_batch_size = sample.size(0)
                optimizer.zero_grad()
                prediction = model(sample)

                if CRITERION == 'CrossEntropyLoss':
                    loss = criterion(prediction, ground)
                elif CRITERION == 'MTLRetinalSelectorLoss':
                    loss = criterion(prediction, ground, dt_name)

                loss.backward()
                optimizer.step()
                if math.isnan(loss.item()):
                    raise ValueError

                running_loss += loss.item() * current_batch_size

                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(current_batch_size)

                if loss.item() <= 0.0001:
                    inf_condition = True

        epoch_loss = running_loss / n_train
        print(f'EPOCH Loss : {epoch_loss}')
        losses.append(epoch_loss)

        # Check train and test datasets to verify train step
        if CONTROL_TRAIN:
            for data_element in ['train', 'test']:
                logging.info(f'....Evaluate [{data_element}] dataset accuracy')
                print(f'    ..........Evaluate "{data_element}" dataset')
                control_dataloader = load_and_transform_data(stage=data_element,
                                                             shuffle=True)
                accuracy = evaluate_model(model,
                                          device,
                                          control_dataloader,
                                          stage=data_element)
                if data_element == 'train':
                    train_accuracy_list.append(accuracy)
                else:
                    test_accuracy_list.append(accuracy)

    return model, (losses, train_accuracy_list, test_accuracy_list)


def evaluate_model(model, device, test_loaders, outer_fold_id, inner_fold_id, max_eval=sys.maxsize, stage='test'):
    """
    Test the model with input parametrization
    :param model: (torch) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param test_loaders: (List torchvision.datasets) List of  train dataloader containing dataset images
    :param outer_fold_id: (int) Outer fold identifier. Just to return data.
    :param inner_fold_id: (int) Inner fold identifier. Just to return data.
    :param max_eval: (int) Maximum number of evaluation samples
    :return: (dict) model accuracy
    """
    n_test = sum([len(test_loaders[i].dataset) for i in range(len(test_loaders))])
    logging.info(f'''Starting MTL testing:
                Test sizes:   {[(test_loaders[i].dataset.dataset_name, len(test_loaders[i].dataset)) for i in range(len(test_loaders))]}
                Device:          {device.type}
            ''')

    correct = 0
    total = 0
    ground_array = []
    prediction_array = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loaders[0]):
            sample, ground, index, dt_name = batch
            sample = sample.to(device=device, dtype=torch.float32)
            ground = ground.to(device=device, dtype=torch.long)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            ground_array.append(ground.item())
            prediction_array.append(predicted.item())

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

            if i > max_eval:
                print(f'Max evaluation reached at: {max_eval} iterations')
                break

    accuracy = 100 * correct / total

    pm = PerformanceMetrics(ground=ground_array,
                            prediction=prediction_array,
                            percent=True,
                            formatted=True)
    confusion_matrix = pm.confusion_matrix()

    performance = {
        f'accuracy_{outer_fold_id}_{inner_fold_id}': pm.accuracy(),
        f'precision_{outer_fold_id}_{inner_fold_id}': pm.precision(),
        f'recall_{outer_fold_id}_{inner_fold_id}': pm.recall(),
        f'f1_{outer_fold_id}_{inner_fold_id}': pm.f1(),
        f'tn_{outer_fold_id}_{inner_fold_id}': confusion_matrix[0],
        f'fp_{outer_fold_id}_{inner_fold_id}': confusion_matrix[1],
        f'fn_{outer_fold_id}_{inner_fold_id}': confusion_matrix[2],
        f'tp_{outer_fold_id}_{inner_fold_id}': confusion_matrix[3]
    }
    return performance, accuracy


if __name__ == '__main__':
    main()
