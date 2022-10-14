"""
# Author: ruben 
# Date: 25/9/22
# Project: MTLFramework
# File: create_folds_structure.py

Description: Create 10-5 fold cross validation structure.
"""
import os
import shutil
import random
from operator import itemgetter

POSITIVE = 'CEP'
NEGATIVE = 'CEN'


def main():
    src = '/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents/working_dataset'

    n_positive_folds = [201, 201, 201, 202, 202]
    n_negative_folds = [103, 103, 103, 103, 103]

    for outer_fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ce_pos_list = os.listdir(os.path.join(src, POSITIVE))
        ce_neg_list = os.listdir(os.path.join(src, NEGATIVE))
        assert len(ce_pos_list) == (201 + 201 + 201 + 202 + 202)
        assert len(ce_neg_list) == (103 + 103 + 103 + 103 + 103)
        tar = f'/home/ruben/PycharmProjects/CardioEvents/input/CE/outer_folds/outer_fold_{outer_fold}'

        for inner_fold_id in [1, 2, 3, 4, 5]:
            # CE Positive folder
            taken_elements = random.sample(range(len(ce_pos_list)), n_positive_folds[inner_fold_id - 1])
            sub_list = itemgetter(*taken_elements)(ce_pos_list)
            ce_pos_list = [elem for elem in ce_pos_list if elem not in sub_list]

            for image in sub_list:
                source = os.path.join(src, POSITIVE, image)
                assert os.path.exists(source)
                target = os.path.join(tar, f'inner_fold_{inner_fold_id}', POSITIVE, image)
                shutil.copy(source, target)
                assert os.path.isfile(os.path.join(target))

            # CE Negative folder
            taken_elements = random.sample(range(len(ce_neg_list)), n_negative_folds[inner_fold_id - 1])
            sub_list = itemgetter(*taken_elements)(ce_neg_list)
            ce_neg_list = [elem for elem in ce_neg_list if elem not in sub_list]

            for image in sub_list:
                source = os.path.join(src, NEGATIVE, image)
                assert os.path.exists(source)
                target = os.path.join(tar, f'inner_fold_{inner_fold_id}', NEGATIVE, image)
                shutil.copy(source, target)
                assert os.path.isfile(os.path.join(target))


if __name__ == '__main__':
    main()
