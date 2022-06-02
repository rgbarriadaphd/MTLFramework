"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: mtl_main.py

Description: Main script to manage MTL framework features
"""
import shutil
import sys
import os
import logging
import datetime
import traceback

from constants.path_constants import *
from constants.train_constants import *

from train_model import TrainModel


def train_model(date_time):
    """
    Initializes train stage
    :param date_time: (str) date and time to identify execution
    """
    tm = TrainModel(date_time=date_time)
    tm.run()


def get_execution_time():
    """
    :return: The current date and time to identify the whole execution
    """
    date_time = datetime.datetime.now()
    return str(date_time.date().strftime('%Y%m%d')) + "_" + str(date_time.time().strftime("%H%M%S"))


def clean_folders():
    """
    Clean logs and models folder
    :return:
    """
    answer_valid = False
    while not answer_valid:
        value = input("[WARNING] Do you want to clear logs and models folders? (y/n):  ")
        if value == 'y' or value == 'yes':
            print("Deleting logs and train storage")
            for f in os.listdir(TRAIN_FOLDER):
                shutil.rmtree(os.path.join(TRAIN_FOLDER, f))
            for f in os.listdir(LOGS_FOLDER):
                os.remove(os.path.join(LOGS_FOLDER, f))
            answer_valid = True
        elif value == 'n' or value == 'no':
            answer_valid = True
        else:
            print(f'Please, "{value}" is not a valid answer. Please, type "y" or "n" ')


def init_log(date_time, action):
    """
    Inits log with specified datetime and action name
    :param date_time: (str) date time
    :param action: (str) action: train, test, exp
    """
    logging.basicConfig(filename=os.path.join(LOGS_FOLDER, f'{action}_{date_time}.log'),
                        level=logging.INFO,
                        format='[%(levelname)s] : %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def display_help():
    """
    Prints usage information
    """
    msg = "MTL Framework\n" \
          "==========================\n" \
          "usage: python mtl_main.py <options>\n" \
          "Options:\n" \
          "  [-h]: Display help.\n" \
          "  [-cl]: Clear logs and models.\n" \
          "  [-tr]: train and test model.\n" \
          "  [-tst] -model=<model_folder>:  test specified model.\n"
    print(msg)


def main(args):
    print(args)
    date_time = get_execution_time()
    try:
        if args[1] == '-tr':
            init_log(date_time, 'train')
            train_model(date_time)
        elif args[1] == '-tst' and '-model' in args[2]:
            init_log(date_time, 'tst')
            # TODO: implement test step over model
        elif args[1] == '-cl':
            clean_folders()
        else:
            display_help()
            return
    except Exception as e:
        logging.error(f'{traceback.format_exc()}')
        return


if __name__ == '__main__':
    main(sys.argv)
