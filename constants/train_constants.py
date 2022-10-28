"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: train_constants.py

Description: Constants regarding train process management
"""

from constants.path_constants import *

# Architecture parameters
# =======================

CUSTOM_NORMALIZED = True  # whether image is normalized retina-based (true) or Imagenet-based (false)
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
REQUIRES_GRAD = False  # Allow backprop in pretrained weights
WEIGHT_INIT = 'Seeded'  # Weight init . Supported --> ['KaimingUniform', 'KaimingNormal', 'XavierUniform', 'XavierNormal']

# Train hyperparameters
# =======================

EPOCHS = 1000
LEARNING_RATE = 0.0001
LR_SCHEDULER = False
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'  # ['CrossEntropyLoss','MTLRetinalSelectorLoss']
OPTIMIZER = 'SDG' # ['SDG', 'ADAM']

# Execution parameters
# =======================
CONTROL_TRAIN = False  # Runs model on train/test every epoch
MONO_FOLD = False  # Run only first Fold (for test)
SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
SAVE_ROC_PLOT = False  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = False  # True if accuracy plot has to be saved
ND = 2  # Number of decimals at outputs

N_INCREASED_FOLDS = [1, 1]
DYNAMIC_FREEZE = False
CLAHE = False

# Datasets parameters
# =======================
# DATASETS = {'CAC': {'batch_size': 8, 'class_values': {'CACSmenos400': 0, 'CACSmas400': 1}, 'path': DYNAMIC_RUN_FOLDER,
#                     'selector': [1, 1, 0, 0, 0, 0, 0],
#                     },
#             'DR': {'batch_size': 12, 'class_values': {
#                 'class_0': 2, 'class_1': 3,
#                 'class_2': 4, 'class_3': 5, 'class_4': 6}, 'path': DR_DATASET_FOLDER,
#                    'selector': [0, 0, 1, 1, 1, 1, 1],
#                    }}

# DATASETS = {'CAC': {'batch_size': 8, 'class_values': {'CACSmenos400': 0, 'CACSmas400': 1}, 'path': DYNAMIC_RUN_FOLDER,
#                     'selector': [1, 1], 'normalization_path': CAC_NORMALIZATION
#                     }}

# Cardiovascular events
normalization = CE_NORMALIZATION_CLAHE if CLAHE else CE_NORMALIZATION_BASELINE
DATASETS = {'CE': {'batch_size': 8, 'class_values': {'CEN': 0, 'CEP': 1}, 'path': DYNAMIC_RUN_FOLDER,
                   'selector': [1, 1], 'normalization_path': normalization
                   }}