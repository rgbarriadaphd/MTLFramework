"""
# Author: ruben 
# Date: 12/10/22
# Project: MTLFramework
# File: roc_plot.py

Description: Computes the ROC AUC for the given measures
"""

"""
# Author: ruben 
# Date: 8/7/22
# Project: MTLFramework
# File: loss_plot.py

Description: Module to plot loss curves
"""
import sys
import os
import csv
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc
import numpy as np

N_CLASSES = 2


def save_plot(plot_title, x_label, y_label, plot_legend, roc_data, output_path):
    """
    Plot ROC AUC curve
    """
    fpr, tpr, roc_auc = roc_data['fpr'], roc_data['tpr'], roc_data['roc_auc']
    print(fpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr["1"],
        tpr["1"],
        color="darkorange",
        lw=lw,
        label=plot_legend.format(roc_auc["1"]),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.savefig(output_path)


def load_json_data(json_file):
    """
    Loads json file containing ROC data
    """
    with open(json_file) as f:
        roc_data = json.load(f)
    return roc_data


if __name__ == '__main__':
    args = sys.argv
    title = args[1]
    label = args[2].split(',')
    x_label, y_label = label[0].strip(), label[1].strip()
    legend = args[3]
    roc_data = load_json_data(args[4])
    output_path = args[5]

    save_plot(title, x_label, y_label, legend, roc_data, output_path)

    # measures = [0.87, 0.91, 0.79, 0.85, 0.8]
    #

    #
    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    #
    # y_test = np.array(
    #     [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
    #      1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    # y_score = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
    #      1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    #
    # print(y_test.shape, y_score.shape)
    # print(type(y_test), type(y_score))
    #
    # for i in range(2):
    #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # print(fpr)
    # print(tpr)
    # print(roc_auc)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(
    #     fpr[1],
    #     tpr[1],
    #     color="darkorange",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f)" % roc_auc[1],
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC")
    # plt.legend(loc="lower right")
    # plt.show()
