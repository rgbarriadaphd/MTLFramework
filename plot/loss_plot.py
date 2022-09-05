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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_plot(plot_title, xy_labels, plot_legend, lenght, y_values, output_path):
    """
    Plot together train and test accuracies by fold
    :param control_accuracies: (tuple) lists of accuracies: test and train for global, CAC and DR
    """

    for i in range(len(y_values)):
        assert len(y_values[i]) == lenght

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xy_labels[0])
    ax1.set_ylabel(xy_labels[1])
    plt.title(plot_title)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    x_epochs = list(range(1, lenght + 1))
    for i in range(len(y_values)):
        ax1.plot(x_epochs, y_values[i], label=plot_legend[i])

    ax1.legend()
    plt.savefig(output_path)


def get_raw_values(csv_files):
    r_files = []
    l = 0
    for csv_file in csv_files:
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            r_files.append([float(elem) for elem in data[0]])
            l = len(data[0])
    return r_files, l


if __name__ == '__main__':
    args = sys.argv

    raw_values, length = get_raw_values(args[4].split(','))

    save_plot(args[1],
              args[2].split(','),
              args[3].split(','),
              length,
              raw_values,
              args[5])
