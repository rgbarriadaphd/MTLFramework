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
from dataset.mtl_dataset import load_and_transform_data


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

    def get(self):
        """
        Return model
        """
        return self._model


class MTLRetinalLoss(nn.Module):
    def __init__(self):
        super(MTLRetinalLoss, self).__init__()
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


def train_model(model, device, cac_train_loader, dr_train_loader):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param cac_train_loader: (torchvision.datasets) CAC Train dataloader containing dataset images
    :param dr_train_loader: (torchvision.datasets) DR Train dataloader containing dataset images
    :param normalization: Normalization to test train dataset
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

    losses = []
    cac_losses = []
    dr_losses = []
    test_global_accuracy_list = []
    test_cac_accuracy_list = []
    test_dr_accuracy_list = []
    train_global_accuracy_list = []
    train_cac_accuracy_list = []
    train_dr_accuracy_list = []

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = MTLRetinalLoss()

    for epoch in range(EPOCHS):

        model.train(True)

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
        epoch_loss = (cac_epoch_loss + dr_epoch_loss) / 2.0

        cac_losses.append(cac_epoch_loss)
        dr_losses.append(dr_epoch_loss)
        losses.append(epoch_loss)

        if SAVE_ACCURACY_PLOT:

            for data_element in ['train', 'test']:
                logging.info(f'....Evaluate [{data_element}] dataset accuracy')
                cac_control_dataloader, dr_control_dataloader = load_and_transform_data(stage=data_element,
                                                                                        batch_size=[1, 1],
                                                                                        shuffle=False)
                global_accuracy, cac_accuracy, dr_accuracy = evaluate_model(model,
                                                                            device,
                                                                            cac_control_dataloader,
                                                                            dr_control_dataloader,
                                                                            max_eval=30)
                if data_element == 'train':
                    train_global_accuracy_list.append(global_accuracy)
                    train_cac_accuracy_list.append(cac_accuracy)
                    train_dr_accuracy_list.append(dr_accuracy)
                else:
                    test_global_accuracy_list.append(global_accuracy)
                    test_cac_accuracy_list.append(cac_accuracy)
                    test_dr_accuracy_list.append(dr_accuracy)

    return model, (cac_losses, dr_losses, losses), (train_global_accuracy_list,
                                                    train_cac_accuracy_list,
                                                    train_dr_accuracy_list,
                                                    test_global_accuracy_list,
                                                    test_cac_accuracy_list,
                                                    test_dr_accuracy_list)


def evaluate_model(model, device, cac_test_loader, dr_test_loader, max_eval=sys.maxsize):
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
    logging.info(f'''Starting testing:
                CAC test size:   {n_cac_test}
                DR test size:   {n_dr_test}
                Device:          {device.type}
            ''')

    correct = 0
    total = 0

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

            cac_total += cac_ground.size(0)
            cac_correct += (predicted == cac_ground).sum().item()

            total += cac_ground.size(0)
            correct += (predicted == cac_ground).sum().item()

            if i > max_eval:
                break

        for i, dr_batch in enumerate(dr_test_loader):

            dr_sample, dr_ground, dr_index = dr_batch
            dr_sample = dr_sample.to(device=device, dtype=torch.float32)
            dr_ground = dr_ground.to(device=device, dtype=torch.long)

            outputs = model(dr_sample)
            _, predicted = torch.max(outputs.data, 1)

            # print(f'DR sample [{dr_index[0]}]Output: {outputs} | Ground: {dr_ground} Predicted: {predicted}')

            dr_total += dr_ground.size(0)
            dr_correct += (predicted == dr_ground).sum().item()

            total += dr_ground.size(0)
            correct += (predicted == dr_ground).sum().item()
            if i > max_eval:
                break

    cac_accuracy = 100 * cac_correct / cac_total
    dr_accuracy = 100 * dr_correct / dr_total
    global_accuracy = (100 * correct) / total
    print(f'CAC Accuracy: {cac_accuracy} %')
    print(f'DR Accuracy: {dr_accuracy} %')
    print(f'Accuracy: {global_accuracy} %')
    return global_accuracy, cac_accuracy, dr_accuracy


if __name__ == '__main__':
    main()
