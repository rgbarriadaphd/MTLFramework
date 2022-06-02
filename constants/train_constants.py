"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: train_constants.py

Description: Constants regarding train process management
"""

# Train hyperparameters
# =======================

EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.001
LR_SCHEDULER = False
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'  # ['CrossEntropyLoss','NegativeLogLikelihood', 'KLDivergence', 'MarginRanking']
OPTIMIZER = 'SDG'


# Architecture parameters
# =======================
CUSTOM_NORMALIZED = False # whether image is normalized retina-based (true) or Imagenet-based (false)
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
REQUIRES_GRAD = True  # Allow backprop in pretrained weights
WEIGHT_INIT = 'Seeded' # Weight init . Supported --> ['KaimingUniform', 'KaimingNormal', 'XavierUniform', 'XavierNormal']

# Output parameters
# =======================
SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = True  # True if accuracy plot has to be saved
ND = 2  # Number of decimals at outputs
