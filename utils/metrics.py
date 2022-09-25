"""
# Author: ruben 
# Date: 1/2/22
# Project: CACFramework
# File: metrics.py

Description: Functions to provide performance metrics
"""
import logging
import statistics
import math

from constants.train_constants import ND


class CrossValidationMeasures:
    """
    Class to get cross validation model performance of all folds
    """

    def __init__(self, measures_list, confidence=1.96, percent=False, formatted=False):
        """
        CrossValidationMeasures constructor
        :param measures_list: (list) List of measures by fold
        :param confidence: (float) Confidence interval percentage
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) whether if data is formatted with 2 decimals or not.
            If activated return string instead of float
        """
        assert (len(measures_list) > 0)
        self._measures = measures_list
        self._confidence = confidence
        self._percent = percent
        self._formatted = formatted
        self._compute_performance()

    def _compute_performance(self):
        """
        Compute mean, std dev and CI of a list of model measures
        """

        if len(self._measures) == 1:
            self._mean = self._measures[0]
            self._stddev = 0.0
            self._offset = 0.0
        else:
            self._mean = statistics.mean(self._measures)
            self._stddev = statistics.stdev(self._measures)
            self._offset = self._confidence * self._stddev / math.sqrt(len(self._measures))
        self._interval = self._mean - self._offset, self._mean + self._offset

    def mean(self):
        """
        :return: Mean value
        """
        if self._percent and self._measures[0] <= 1.0:
            mean = self._mean * 100.0
        else:
            mean = self._mean
        if self._formatted:
            return f'{mean:.{ND}f}'
        else:
            return mean

    def stddev(self):
        """
        :return: Std Dev value
        """
        if self._percent and self._measures[0] <= 1.0:
            stddev = self._stddev * 100.0
        else:
            stddev = self._stddev
        if self._formatted:
            return f'{stddev:.{ND}f}'
        else:
            return stddev

    def interval(self):
        """
        :return: Confidence interval
        """
        if self._percent:
            interval = self._interval[0] * 100.0, self._interval[1] * 100.0
        else:
            interval = self._interval[0], self._interval[1]
        if self._formatted:
            return f'({self._interval[0]:.{ND}f}, {self._interval[1]:.{ND}f})'
        else:
            return interval


class PerformanceMetrics:
    """
    Class to compute model performance
    """

    def __init__(self, ground, prediction, percent=False, formatted=False):
        """
        PerformanceMetrics class constructor
        :param ground: input array of ground truth
        :param prediction: input array of prediction values
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) whether if data is formatted with 2 decimals or not.
            If activated return string instead of float
        """
        assert (len(ground) == len(prediction))
        self._ground = ground
        self._prediction = prediction
        self._percent = percent
        self._formatted = formatted
        self._confusion_matrix = None
        self._accuracy = None
        self._precision = None
        self._recall = None
        self._f1 = None
        self._compute_measures()

    def _compute_measures(self):
        """
        Compute performance measures
        """
        self._compute_confusion_matrix()
        self._compute_accuracy()
        self._compute_precision()
        self._compute_recall()
        self._compute_f1()

    def _compute_confusion_matrix(self):
        """
        Computes the confusion matrix of a model
        """
        self._tp, self._fp, self._tn, self._fn = 0, 0, 0, 0

        for i in range(len(self._prediction)):
            if self._ground[i] == self._prediction[i] == 1:
                self._tp += 1
            if self._prediction[i] == 1 and self._ground[i] != self._prediction[i]:
                self._fp += 1
            if self._ground[i] == self._prediction[i] == 0:
                self._tn += 1
            if self._prediction[i] == 0 and self._ground[i] != self._prediction[i]:
                self._fn += 1

        self._confusion_matrix = self._tn, self._fp, self._fn, self._tp

    def _compute_accuracy(self):
        """
        Computes the accuracy of a model
        """
        self._accuracy = (self._tn + self._tp) / len(self._prediction)

    def _compute_precision(self):
        """
        Computes the precision of a model
        """
        try:
            self._precision = self._tp / (self._tp + self._fp)
        except ZeroDivisionError:
            self._precision = 0.0

    def _compute_recall(self):
        """
        Computes the recall of a model
        """
        try:
            self._recall = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            self._recall = 0.0

    def _compute_f1(self):
        """
        Computes the F1 measure of a model
        """
        try:
            self._f1 = 2 * (self._precision * self._recall / (self._precision + self._recall))
        except ZeroDivisionError:
            self._f1 = 0.0

    def confusion_matrix(self):
        """
        :return: Confusion matrix
        """
        return self._confusion_matrix

    def accuracy(self):
        """
        :return: Accuracy measure
        """
        if self._percent:
            accuracy = self._accuracy * 100.0
        else:
            accuracy = self._accuracy
        if self._formatted:
            return f'{accuracy:.{ND}f}'
        else:
            return accuracy

    def precision(self):
        """
        :return: Precision measure
        """
        if self._percent:
            precision = self._precision * 100.0
        else:
            precision = self._precision
        if self._formatted:
            return f'{precision:.{ND}f}'
        else:
            return precision

    def recall(self):
        """
        :return: Recall measure
        """
        if self._percent:
            recall = self._recall * 100.0
        else:
            recall = self._recall
        if self._formatted:
            return f'{recall:.{ND}f}'
        else:
            return recall

    def f1(self):
        """
        :return: F1 measure
        """
        if self._percent:
            f1 = self._f1 * 100.0
        else:
            f1 = self._f1
        if self._formatted:
            return f'{f1:.{ND}f}'
        else:
            return f1


if __name__ == '__main__':
    # Test functions
    mground = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    mprediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    pm = PerformanceMetrics(mground, mprediction, percent=True, formatted=True)
    conf_matrix = pm.confusion_matrix()

    assert conf_matrix[0] == 17
    assert conf_matrix[1] == 2
    assert conf_matrix[2] == 3
    assert conf_matrix[3] == 8

    print(f'TN: {conf_matrix[0]}')
    print(f'FP: {conf_matrix[1]}')
    print(f'FN: {conf_matrix[2]}')
    print(f'TP: {conf_matrix[3]}')

    print(f'Accuracy: {pm.accuracy()}')
    print(f'Recall: {pm.recall()}')
    print(f'Precision: {pm.precision()}')
    print(f'F1-measure: {pm.f1()}')

    measures = [0.51, 0.45, 0.78, 0.79, 0.82]
    cvm = CrossValidationMeasures(measures, percent=True, formatted=True)

    print(f'Mean: {cvm.mean()}')
    print(f'Std Dev: {cvm.stddev()}')
    print(f'Interval: {cvm.interval()}')
