"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: path_constants.py

Description: Constants regarding path management
"""

import os

# Main folder structure
# =======================
OUTPUT_FOLDER = 'output/'
assert os.path.exists(OUTPUT_FOLDER)

INPUT_FOLDER = 'input/'
assert os.path.exists(OUTPUT_FOLDER)

LOGS_FOLDER = os.path.join(OUTPUT_FOLDER, 'log')
assert os.path.exists(LOGS_FOLDER)
TRAIN_FOLDER = os.path.join(OUTPUT_FOLDER, 'train')
assert os.path.exists(TRAIN_FOLDER)

CAC_DATASET_FOLDER = os.path.join(INPUT_FOLDER, 'CAC')
assert os.path.exists(CAC_DATASET_FOLDER)
DR_DATASET_FOLDER = os.path.join(INPUT_FOLDER, 'DR')
assert os.path.exists(DR_DATASET_FOLDER)

# Templates
# =======================
SUMMARY_TEMPLATE = 'templates/summary.tpl'
assert os.path.exists(SUMMARY_TEMPLATE)