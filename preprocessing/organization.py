"""
# Author: ruben 
# Date: 25/10/22
# Project: MTLFramework
# File: organization.py

Description: Save outer fold image organization to a json file
"""
import json
import os


def save_json(param_dict, json_path):
    with open(json_path, 'w') as f:
        json.dump(param_dict, f)


if __name__ == '__main__':
    fold_path = '/home/ruben/PycharmProjects/MTLFramework/input/CE/outer_folds'
    out_json = '/home/ruben/PycharmProjects/MTLFramework/input/CE/organization.json'

    org = {}
    for i in range(1, 11):
        org[i] = {}
        for j in range(1, 6):
            org[i][j] = {'CEP': None, 'CEN': None}
            src_cen = os.path.join(fold_path, f'outer_fold_{i}', f'inner_fold_{j}', 'CEN')
            src_cep = os.path.join(fold_path, f'outer_fold_{i}', f'inner_fold_{j}', 'CEP')
            assert os.path.exists(src_cen)
            assert os.path.exists(src_cep)
            org[i][j]['CEN'] = os.listdir(src_cen)
            org[i][j]['CEP'] = os.listdir(src_cep)

    save_json(org, out_json)
