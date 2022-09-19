"""
# Author: ruben 
# Date: 19/9/22
# Project: MTLFramework
# File: generate_outter_folds.py

Description: "Enter feature description here"
"""
import os
import shutil
import random
from operator import itemgetter

CACMAS = 'CACSmas400'
CACMENOS = 'CACSmenos400'


def main():
    src = '/home/ruben/PycharmProjects/MTLFramework/input/CAC/raw'

    n_mas_folds = [14, 14, 14, 14, 12]
    n_menos_folds = [18, 18, 18, 16, 16]


    for outter_fold in [4, 5, 6, 7, 8, 9, 10]:
        cacmas_list = os.listdir(os.path.join(src, CACMAS))
        cacmenos_list = os.listdir(os.path.join(src, CACMENOS))
        assert len(cacmas_list) == (14 + 14 + 14 + 14 + 12)
        assert len(cacmenos_list) == (18 + 18 + 18 + 16 + 16)
        tar = f'/home/ruben/PycharmProjects/MTLFramework/input/CAC/outter_folds/outter_fold_{outter_fold}'
        for inner_fold_id in [1, 2, 3, 4, 5]:

            # CAC MAS folder
            taken_elements = random.sample(range(len(cacmas_list)), n_mas_folds[inner_fold_id - 1])
            sub_list = itemgetter(*taken_elements)(cacmas_list)
            cacmas_list = [elem for elem in cacmas_list if elem not in sub_list]

            for image in sub_list:
                source = os.path.join(src, CACMAS, image)
                assert os.path.exists(source)
                target = os.path.join(tar, f'inner_fold_{inner_fold_id}', CACMAS, image)
                shutil.copy(source, target)
                assert os.path.isfile(os.path.join(target))

            # CAC MENOS folder
            taken_elements = random.sample(range(len(cacmenos_list)), n_menos_folds[inner_fold_id - 1])
            sub_list = itemgetter(*taken_elements)(cacmenos_list)
            cacmenos_list = [elem for elem in cacmenos_list if elem not in sub_list]

            for image in sub_list:
                source = os.path.join(src, CACMENOS, image)
                assert os.path.exists(source)
                target = os.path.join(tar, f'inner_fold_{inner_fold_id}', CACMENOS, image)
                shutil.copy(source, target)
                assert os.path.isfile(os.path.join(target))



if __name__ == '__main__':
    main()
