"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: train_constants.py

Description: Constants regarding train process management
"""

# Architecture parameters
# =======================
from constants.path_constants import CAC_DATASET_FOLDER, DR_DATASET_FOLDER, DYNAMIC_RUN_FOLDER

DATASETS = {'CAC': {'batch_size': 8, 'class_values': {'CACSmenos400': 0, 'CACSmas400': 1}, 'path': DYNAMIC_RUN_FOLDER,
                    'selector': [1, 1, 0, 0, 0, 0, 0],
                    'normalization': {
                        'mean': [0.479, 0.241, 0.126],
                        'std': [0.268, 0.142, 0.079]
                    }},
            'DR': {'batch_size': 12, 'class_values': {
                'class_0': 2, 'class_1': 3,
                'class_2': 4, 'class_3': 5, 'class_4': 6}, 'path': DR_DATASET_FOLDER,
                   'selector': [0, 0, 1, 1, 1, 1, 1],
                   'normalization': {
                       'mean': [0.439, 0.304, 0.216],
                       'std': [0.200, 0.139, 0.099]
                   }
                   }}
# DATASETS = {'CAC': {'batch_size': 8, 'class_values': {'CACSmenos400': 0, 'CACSmas400': 1}, 'path': CAC_DATASET_FOLDER,
#                     'selector': [1, 1],
#                     'normalization': {
#                         'mean': [0.479, 0.241, 0.126],
#                         'std': [0.268, 0.142, 0.079]
#                     }}}

CUSTOM_NORMALIZED = True  # whether image is normalized retina-based (true) or Imagenet-based (false)
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
REQUIRES_GRAD = True  # Allow backprop in pretrained weights
WEIGHT_INIT = 'Seeded'  # Weight init . Supported --> ['KaimingUniform', 'KaimingNormal', 'XavierUniform', 'XavierNormal']

# Train hyperparameters
# =======================

EPOCHS = 100
LEARNING_RATE = 0.0001
LR_SCHEDULER = False
WEIGHT_DECAY = 4e-2
CRITERION = 'MTLRetinalSelectorLoss'  # ['CrossEntropyLoss','MTLRetinalSelectorLoss']
OPTIMIZER = 'SDG'

# Execution parameters
# =======================

CONTROL_TRAIN = False  # Runs model on train/test every epoch
SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = True  # True if accuracy plot has to be saved
ND = 2  # Number of decimals at outputs
