"""
# Author: ruben 
# Date: 30/5/22
# Project: MTLFramework
# File: cnn.py

Description: Functions to deal with the cnn operations
"""
import logging
from itertools import cycle
import sys

import torch
from torch import nn
from torchvision import models
from torch import optim
from tqdm import tqdm
from constants.train_constants import *
from dataset.base_dataset import load_and_transform_base_data
from dataset.mtl_dataset import load_and_transform_mtl_data


def main():
    print("run")


class DLModel:
    """
    Class to manage the architecture initialization
    """

    def __init__(self, device, path=None, n_classes=2, load_model=False):
        """
        Architecture class constructor
        :param device: (torch.device) Running device
        :param path: (str) If defined, then the model has to be loaded.
        :param n_classes: (int) number of classes.
        """
        self._model_path = path
        self._device = device
        self._model = None

        if MODEL_SEED > 0:
            torch.manual_seed(MODEL_SEED)

        if load_model:
            self._model = models.vgg16()
            num_features = self._model.classifier[6].in_features
            features = list(self._model.classifier.children())[:-1]  # Remove last layer
            linear = nn.Linear(num_features, 5) # It is a DR model
            features.extend([linear])
            self._model.classifier = nn.Sequential(*features)
            self._model.load_state_dict(torch.load(MODEL_PATH))
        else:
            self._model = models.vgg16(pretrained=True)

        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, n_classes)
        features.extend([linear])
        self._model.classifier = nn.Sequential(*features)

        for param in self._model.parameters():
            param.requires_grad = REQUIRES_GRAD

        if self._model_path:
            self._model.load_state_dict(torch.load(self._model_path, map_location=torch.device(self._device)))
            logging.info(f'Loading architecture from {self._model_path}')

    def get(self):
        """
        Return model
        """
        return self._model


class MTLRetinalSelectorLoss(nn.Module):
    def __init__(self):
        super(MTLRetinalSelectorLoss, self).__init__()
        self._selector = 0

    def forward(self, preds, cac, dr):
        self._selector = 0 if preds[1] is None else 1

        if self._selector == 0:
            assert preds[self._selector] is not None

            cac_cross_entropy = nn.CrossEntropyLoss()
            cac_loss = cac_cross_entropy(preds[0], cac)
            return cac_loss
        else:
            assert preds[self._selector] is not None
            dr_cross_entropy = nn.CrossEntropyLoss()
            dr_loss = dr_cross_entropy(preds[1], dr)
            return dr_loss

class MTLRetinalSumLoss(nn.Module):
    def __init__(self):
        super(MTLRetinalSumLoss, self).__init__()
        self._selector = 0

    def forward(self, preds, cac, dr):
        self._selector = 0 if preds[1] is None else 1

        if self._selector == 0:
            assert preds[self._selector] is not None

            cac_cross_entropy = nn.CrossEntropyLoss()
            cac_loss = cac_cross_entropy(preds[0], cac)
            return cac_loss
        else:
            assert preds[self._selector] is not None
            dr_cross_entropy = nn.CrossEntropyLoss()
            dr_loss = dr_cross_entropy(preds[1], dr)
            return dr_loss


def train_mtl_model(model, device, cac_train_loader, dr_train_loader):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param cac_train_loader: (torchvision.datasets) CAC Train dataloader containing dataset images
    :param dr_train_loader: (torchvision.datasets) DR Train dataloader containing dataset images
    :return: train model, losses array, accuracies of test and train datasets
    """
    n_cac_train = len(cac_train_loader.dataset)
    n_dr_train = len(dr_train_loader.dataset)
    logging.info(f'''Starting training:
            Epochs:          {EPOCHS}
            Batch size:      {BATCH_SIZE}
            Learning rate:   {LEARNING_RATE}
            CAC Training size:   {n_cac_train}
            DR Training size:   {n_dr_train}
            Device:          {device.type}
            Criterion:       {CRITERION}
            Optimizer:       {OPTIMIZER}
        ''')

    cac_losses = []
    dr_losses = []
    test_cac_accuracy_list = []
    test_dr_accuracy_list = []
    train_cac_accuracy_list = []
    train_dr_accuracy_list = []

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if LOSS_CRITERION == 'selector_loss':
        criterion = MTLRetinalSelectorLoss()
    elif LOSS_CRITERION == 'sum_loss':
        criterion = MTLRetinalSumLoss()

    for epoch in range(EPOCHS):

        model.train(True)
        print(f'=================================== {epoch + 1} ===================================')

        with tqdm(total=n_cac_train + n_dr_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:

            cac_running_loss = 0.0
            dr_running_loss = 0.0
            for i, (cac_batch, dr_batch) in enumerate(zip(cycle(cac_train_loader), dr_train_loader)):
                # CAC batch ..................................................
                cac_sample, cac_ground, cac_index = cac_batch
                cac_sample = cac_sample.to(device=device, dtype=torch.float32)
                cac_ground = cac_ground.to(device=device, dtype=torch.long)

                current_cac_batch_size = cac_sample.size(0)
                optimizer.zero_grad()
                cac_prediction = model(cac_sample)

                cac_loss = criterion((cac_prediction, None), cac_ground, None)
                cac_loss.backward()
                optimizer.step()
                cac_running_loss += cac_loss.item() * current_cac_batch_size

                # DR batch ..................................................
                dr_sample, dr_ground, dr_index = dr_batch
                dr_sample = dr_sample.to(device=device, dtype=torch.float32)
                dr_ground = dr_ground.to(device=device, dtype=torch.long)

                current_dr_batch_size = dr_sample.size(0)
                optimizer.zero_grad()
                dr_prediction = model(dr_sample)

                dr_loss = criterion((None, dr_prediction), None, dr_ground)
                dr_loss.backward()
                optimizer.step()
                dr_running_loss += dr_loss.item() * current_dr_batch_size

                pbar.set_postfix(**{'CAC loss (batch) ': cac_loss.item(),
                                    'DR loss (batch) ': dr_loss.item()})
                pbar.update(current_cac_batch_size + current_dr_batch_size)

        cac_epoch_loss = cac_running_loss / n_cac_train
        dr_epoch_loss = dr_running_loss / n_dr_train

        cac_losses.append(cac_epoch_loss)
        dr_losses.append(dr_epoch_loss)

        if SAVE_ACCURACY_PLOT:

            for data_element in ['train', 'test']:
                logging.info(f'....Evaluate [{data_element}] dataset accuracy')
                print(f'    ..........Evaluate "{data_element}" dataset')
                cac_control_dataloader, dr_control_dataloader = load_and_transform_mtl_data(stage=data_element,
                                                                                            batch_size=[1, 1],
                                                                                            shuffle=True)
                cac_accuracy, dr_accuracy = evaluate_mtl_model(model,
                                                               device,
                                                               cac_control_dataloader,
                                                               dr_control_dataloader,
                                                               stage=data_element)
                print(f'    ..........CAC acc.: {cac_accuracy}')
                print(f'    ..........DR acc.: {dr_accuracy}')
                if data_element == 'train':
                    train_cac_accuracy_list.append(cac_accuracy)
                    train_dr_accuracy_list.append(dr_accuracy)
                else:
                    test_cac_accuracy_list.append(cac_accuracy)
                    test_dr_accuracy_list.append(dr_accuracy)

    return model, (cac_losses, dr_losses), (train_cac_accuracy_list,
                                            train_dr_accuracy_list,
                                            test_cac_accuracy_list,
                                            test_dr_accuracy_list)


def evaluate_mtl_model(model, device, cac_test_loader, dr_test_loader, max_eval=sys.maxsize, stage='test'):
    """
    Test the model with input parametrization
    :param model: (torch) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param cac_test_loader: (torchvision.datasets) CAC Test dataloader containing dataset images
    :param dr_test_loader: (torchvision.datasets) DR Train dataloader containing dataset images
    :param max_eval: (int) Maximum number of evaluation samples
    :return: (dict) model performance including accuracy, precision, recall, F1-measure
            and confusion matrix
    """
    n_cac_test = len(cac_test_loader.dataset)
    n_dr_test = len(dr_test_loader.dataset)
    logging.info(f'''Starting MTL testing:
                CAC {stage} size:   {n_cac_test}
                DR {stage} size:   {n_dr_test}
                Device:          {device.type}
            ''')

    cac_correct = 0
    cac_total = 0
    dr_correct = 0
    dr_total = 0

    model.eval()
    with torch.no_grad():
        for i, cac_batch in enumerate(cac_test_loader):

            cac_sample, cac_ground, cac_index = cac_batch
            cac_sample = cac_sample.to(device=device, dtype=torch.float32)
            cac_ground = cac_ground.to(device=device, dtype=torch.long)

            outputs = model(cac_sample)
            _, predicted = torch.max(outputs.data, 1)

            # print(f'CAC sample [{cac_index}]Output: {outputs} | Ground: {cac_ground} Predicted: {predicted}')
            # print(f'DR [{i}] Ground: {cac_ground.item()} Predicted: {predicted.item()}')

            cac_total += cac_ground.size(0)
            cac_correct += (predicted == cac_ground).sum().item()

            if i > max_eval:
                print(f'Max evaluatoin reached at: {max_eval} iterations')
                break

        for i, dr_batch in enumerate(dr_test_loader):

            dr_sample, dr_ground, dr_index = dr_batch
            dr_sample = dr_sample.to(device=device, dtype=torch.float32)
            dr_ground = dr_ground.to(device=device, dtype=torch.long)

            outputs = model(dr_sample)
            _, predicted = torch.max(outputs.data, 1)

            # print(f'DR sample [{dr_index[0]}]Output: {outputs} | Ground: {dr_ground} Predicted: {predicted}')
            # print(f'DR [{i}] Ground: {dr_ground.item()} Predicted: {predicted.item()}')
            dr_total += dr_ground.size(0)
            dr_correct += (predicted == dr_ground).sum().item()

            if i > max_eval:
                print(f'Max evaluatoin reached at: {max_eval} iterations')
                break

    cac_accuracy = 100 * cac_correct / cac_total
    dr_accuracy = 100 * dr_correct / dr_total

    return cac_accuracy, dr_accuracy



def train_base_model(model, device, train_loader):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param train_loader: (torchvision.datasets) CAC Train dataloader containing dataset images
    :return: train model, losses array, accuracies of test and train datasets
    """
    n_train = len(train_loader.dataset)

    logging.info(f'''Starting training:
            Epochs:          {EPOCHS}
            Batch size:      {BATCH_SIZE}
            Learning rate:   {LEARNING_RATE}
            BASE Training size:   {n_train}
            Device:          {device.type}
            Criterion:       {CRITERION}
            Optimizer:       {OPTIMIZER}
        ''')

    losses = []
    train_accuracies = []
    test_accuracies = []

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train(True)
        running_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:

            cac_running_loss = 0.0
            dr_running_loss = 0.0
            for i, batch in enumerate(train_loader):
                sample, ground, index = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                current_batch_size = sample.size(0)
                optimizer.zero_grad()
                prediction = model(sample)

                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * current_batch_size

                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(sample.shape[0])

        epoch_loss = running_loss / n_train
        losses.append(epoch_loss)


        if SAVE_ACCURACY_PLOT:

            for data_element in ['train', 'test']:
                logging.info(f'....Evaluate [{data_element}] dataset accuracy')
                dataloader = load_and_transform_base_data(stage=data_element,
                                                          batch_size=1,
                                                          shuffle=False)

                accuracy = evaluate_base_model(model,
                                               device,
                                               dataloader)
                if data_element == 'train':
                    train_accuracies.append(accuracy)
                else:
                    test_accuracies.append(accuracy)

    return model, losses, (train_accuracies, test_accuracies)


def evaluate_base_model(model, device, test_loader):
    """
    Test the model with input parametrization
    :param model: (torch) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param test_loader: (torchvision.datasets) CAC Test dataloader containing dataset images
    :return: (dict) model performance including accuracy, precision, recall, F1-measure
            and confusion matrix
    """
    n_test = len(test_loader.dataset)

    logging.info(f'''Starting testing:
                BASE test size:   {n_test}
                Device:          {device.type}
            ''')

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            sample, ground, _ = batch
            sample = sample.to(device=device, dtype=torch.float32)
            ground = ground.to(device=device, dtype=torch.long)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            # print(f'CAC sample [{cac_index}]Output: {outputs} | Ground: {cac_ground} Predicted: {predicted}')

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

    accuracy = (100 * correct) / total

    print(f'BASE Accuracy: {accuracy} %')
    return accuracy

if __name__ == '__main__':
    main()
