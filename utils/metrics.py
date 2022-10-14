"""
# Author: ruben
# Date: 1/2/22
# Project: CACFramework
# File: metrics.py

Description: Functions to provide performance metrics
"""
import logging
import random
import statistics
import math
from sklearn.metrics import roc_curve, auc
import numpy as np

ND = 2


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
        self._fpr = dict()
        self._tpr = dict()
        self._roc_auc = dict()
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
        self._compute_roc_params()

    def _compute_roc_params(self):
        """
        Computes ROC AUC parameters
        """
        ground = np.array(self._ground)
        prediction = np.array(self._prediction)
        for i in range(2):
            self._fpr[i], self._tpr[i], _ = roc_curve(ground, prediction)
            self._roc_auc[i] = auc(self._fpr[i], self._tpr[i])
            self._fpr[i] = list(self._fpr[i])
            self._tpr[i] = list(self._tpr[i])

        # Compute micro-average ROC curve and ROC area
        self._fpr["micro"], self._tpr["micro"], _ = roc_curve(ground.ravel(), prediction.ravel())
        self._roc_auc["micro"] = auc(self._fpr["micro"], self._tpr["micro"])
        self._fpr["micro"] = list(self._fpr["micro"])
        self._tpr["micro"] = list(self._tpr["micro"])

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

    def fpr(self):
        """
        :return: False positive rate for ROC
        """
        return self._fpr

    def tpr(self):
        """
        :return: True positive rate for ROC
        """
        return self._tpr

    def roc_auc(self):
        """
        :return:ROC AUC
        """
        return self._roc_auc


def global_results(measures):
    cvm = CrossValidationMeasures(measures, percent=True, formatted=True)
    mean = cvm.mean()
    stddev = cvm.stddev()
    ci = cvm.interval()
    print('\\begin{figure}[H]')
    print('\centering')
    print("$Acc=\\begin{pmatrix}")
    sacc = ''
    for pos, elem in enumerate(measures):

        if (pos + 1) % 5 == 0:
            sacc += f'{elem:.{ND}f} \\\\'
            print(sacc)
            sacc = ''
        else:
            sacc += f'{elem:.{ND}f} & '

    print("\end{pmatrix}$")
    print('\end{figure}')
    print()
    print()
    print("\\begin{table}[H]")
    print('\centering')
    print("\\begin{tabular}{|c|c|}")
    print("	\hline")
    print(f'	\\textbf{{Mean}} & {mean} \\\\')
    print("	\hline")
    print(f'	\\textbf{{Std Dev}} &  {stddev}\\\\')
    print("	\hline")
    print(f'	\\textbf{{CI(95\%)}} & {ci}\\\\')
    print("	\hline")
    print("\end{tabular}")
    print("\label{tab:metrics}")
    print("\end{table}")


def by_outer_fold_results(measures):
    print('\\begin{table}[H]')
    print('\centering')
    print('\\begin{tabular}{|c|c|c|c|}')
    print('\hline')
    print('\\textbf{Fold ID} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{CI(95\%)}\\\\')
    print('\hline')
    acc = []
    fold = 0
    for pos, elem in enumerate(measures):
        if (pos + 1) % 5 == 0:
            fold += 1
            acc.append(elem)
            cvm = CrossValidationMeasures(acc, percent=True, formatted=True)
            mean = cvm.mean()
            stddev = cvm.stddev()
            ci = cvm.interval()
            print(f'\\textbf{{1-{fold}}} & {mean} & {stddev} & ${ci}$ \\\\')
            print('\hline')
            acc = []
        else:
            acc.append(elem)
    print('\end{tabular}')
    print('\end{table}')


def comparison_global(measures_a, measures_b):
    cvm_a = CrossValidationMeasures(measures_a, percent=True, formatted=True)
    mean_a = cvm_a.mean()
    stddev_a = cvm_a.stddev()
    ci_a = cvm_a.interval()
    cvm_b = CrossValidationMeasures(measures_b, percent=True, formatted=True)
    mean_b = cvm_b.mean()
    stddev_b = cvm_b.stddev()
    ci_b = cvm_b.interval()
    print('\\begin{table}[H]')
    print('\centering')
    print('\\begin{tabular}{|c|c|c|}')
    print('\hline')
    print('&  \\textbf{MTL} &  \\textbf{CAC}\\\\')
    print('\hline')
    print(f'\\textbf{{Mean}} & {mean_a} & {mean_b} \\\\')
    print('\hline')
    print(f'\\textbf{{Std Dev}} & {stddev_a} & {stddev_b} \\\\')
    print('\hline')
    print(f'\\textbf{{CI(95\%)}} & ${ci_a}$ & ${ci_b}$ \\\\')
    print('\hline')
    print('\end{tabular}')
    print('\end{table}')


def comparison_by_outer_fold_results(measures_a, measures_b):
    """


\multirow{2}{*}{\textbf{1-5}} & MTL & 6 & 230 & 12\\
& CAC & 5 & 195 & 12\\
\hline
\hline
\multirow{2}{*}{\textbf{1-5}} & MTL & 6 & 230 & 12\\
& CAC & 5 & 195 & 12\\
\hline
\end{tabular}
\end{table}
    """
    print('\\begin{table}[H]')
    print('\centering')
    print('\\begin{tabular}{|c|c|c|c|c|}')
    print('\hline')
    print('\\textbf{Fold ID} & \\textbf{Model} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{CI(95\%)}\\\\')
    print('\hline')

    acc_a = []
    acc_b = []
    fold = 0
    for pos, (a, b) in enumerate(zip(measures_a, measures_b)):
        if (pos + 1) % 5 == 0:
            fold += 1
            acc_a.append(a)
            acc_b.append(b)
            cvm_a = CrossValidationMeasures(acc_a, percent=True, formatted=True)
            mean_a = cvm_a.mean()
            stddev_a = cvm_a.stddev()
            ci_a = cvm_a.interval()
            cvm_b = CrossValidationMeasures(acc_b, percent=True, formatted=True)
            mean_b = cvm_b.mean()
            stddev_b = cvm_b.stddev()
            ci_b = cvm_b.interval()

            print(f'\multirow{{2}}{{*}}{{\\textbf{{1-{fold}}}}} & MTL & {mean_a} & {stddev_a} & ${ci_a}$\\\\')
            print(f'& CAC & {mean_b} & {stddev_b} & ${ci_b}$\\\\')
            print('\hline')
            print('\hline')

            acc_a = []
            acc_b = []
        else:
            acc_a.append(a)
            acc_b.append(b)

    print('\end{tabular}')
    print('\end{table}')


if __name__ == '__main__':
    # # Test functions
    # mground = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # mprediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    # pm = PerformanceMetrics(mground, mprediction, percent=True, formatted=True)
    # conf_matrix = pm.confusion_matrix()
    #
    # assert conf_matrix[0] == 17
    # assert conf_matrix[1] == 2
    # assert conf_matrix[2] == 3
    # assert conf_matrix[3] == 8
    #
    # print(f'TN: {conf_matrix[0]}')
    # print(f'FP: {conf_matrix[1]}')
    # print(f'FN: {conf_matrix[2]}')
    # print(f'TP: {conf_matrix[3]}')
    #
    # print(f'Accuracy: {pm.accuracy()}')
    # print(f'Recall: {pm.recall()}')
    # print(f'Precision: {pm.precision()}')
    # print(f'F1-measure: {pm.f1()}')
    #
    # measures = [0.51, 0.45, 0.78, 0.79, 0.82]
    # cvm = CrossValidationMeasures(measures, percent=True, formatted=True)
    #
    # print(f'Mean: {cvm.mean()}')
    # print(f'Std Dev: {cvm.stddev()}')
    # print(f'Interval: {}')

    # mtl_measures = [random.uniform(0, 1) for i in range(50)]
    # cac_measures = [random.uniform(0, 1) for i in range(50)]

    # print('\section{10-5 Cross Validation}')
    # print('\subsection{MTL}')
    # print('\subsubsection{Global Results}')
    # global_results(mtl_measures)
    # print('\subsubsection{By outer fold}')
    # by_outer_fold_results(mtl_measures)
    # print('\subsection{CAC}')
    # print('\subsubsection{Global Results}')
    # global_results(cac_measures)
    # print('\subsubsection{By outer fold}')
    # by_outer_fold_results(cac_measures)
    # print('\subsection{Comparison MTL vs. CAC}')
    # print('\subsubsection{Global Results}')
    # comparison_global(mtl_measures, cac_measures)
    # print('\subsubsection{By outer fold}')
    # comparison_by_outer_fold_results(mtl_measures, cac_measures)


    mground = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    mprediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    pm = PerformanceMetrics(mground, mprediction, percent=True, formatted=True)
    a = 6