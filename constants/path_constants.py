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

CE_DATASET_FOLDER = os.path.join(INPUT_FOLDER, 'CE')
assert os.path.exists(CE_DATASET_FOLDER)

DYNAMIC_RUN_FOLDER = os.path.join(CE_DATASET_FOLDER, 'dynamic_run')
assert os.path.exists(DYNAMIC_RUN_FOLDER)

ROOT_ORIGINAL_FOLDS = os.path.join(CE_DATASET_FOLDER, 'outer_folds_baseline')
assert os.path.exists(ROOT_ORIGINAL_FOLDS)

ROOT_ORIGINAL_FOLDS_CLAHE = os.path.join(CE_DATASET_FOLDER, 'outer_folds_clahe')
assert os.path.exists(ROOT_ORIGINAL_FOLDS_CLAHE)

CAC_NEGATIVE = 'CACSmenos400'
CAC_POSITIVE = 'CACSmas400'
CEP = 'CEP'
CEN = 'CEN'
TRAIN = 'train'
TEST = 'test'

# Templates
# =======================
BASE_SUMMARY_TEMPLATE = 'templates/summary_base.tpl'
SUMMARY_TEMPLATE = 'templates/summary.tpl'
SUMMARY_UNIQUE_FOLD_TEMPLATE = 'templates/fold.tpl'

# Output Folder
# =======================
CSV_FOLDER = 'csv'
PLOT_FOLDER = 'plots'
FOLD_FOLDER = 'folds'

# Normalization files
# =======================
CAC_NORMALIZATION = os.path.join(CAC_DATASET_FOLDER, 'normalization.json')
CE_NORMALIZATION = os.path.join(CE_DATASET_FOLDER, 'normalization.json')
CE_NORMALIZATION_CLAHE = os.path.join(CE_DATASET_FOLDER, 'normalization_clahe.json')
CE_NORMALIZATION_BASELINE = os.path.join(CE_DATASET_FOLDER, 'normalization_baseline.json')