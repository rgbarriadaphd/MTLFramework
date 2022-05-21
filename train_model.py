"""
# Author: ruben 
# Date: 21/5/22
# Project: MTLFramework
# File: train_model.py

Description: Class to handle train stages
"""


class TrainModel:

    def __init__(self, date_time):
        """
        Model train constructor initializes al class attributes.
        :param date_time: (str) date and time to identify execution
        """
        self._date_time = date_time

    def run(self):
        """
        Run train stage
        """
        print('run')