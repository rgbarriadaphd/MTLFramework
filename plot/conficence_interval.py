"""
# Author: ruben 
# Date: 26/9/22
# Project: MTLFramework
# File: conficence_interval.py

Description: Module to plot confidence intervals
"""

import matplotlib.pyplot as plt
import statistics
from math import sqrt
import numpy as np
from pylab import *
import random


def plot_confidence_interval(x, values_a, values_b, z=1.96, horizontal_line_width=0.25, legend=False, text=None):
    # Plot first interval
    color_a = 'grey'
    mean_a = statistics.mean(values_a)
    stdev_a = statistics.stdev(values_a)
    confidence_interval_a = z * stdev_a / sqrt(len(values_a))

    left = x - horizontal_line_width / 2
    top = mean_a - confidence_interval_a
    right = x + horizontal_line_width / 2
    bottom = mean_a + confidence_interval_a

    plt.plot([x, x], [top, bottom], color=color_a)
    plt.plot([left, right], [top, top], color=color_a)
    plt.plot([left, right], [bottom, bottom], color=color_a)
    if legend:
        plt.plot(x, mean_a, 'o', color='red', label=text[0])
    else:
        plt.plot(x, mean_a, 'o', color='red')

    # Plot second interval
    color_b = 'sienna'
    mean_b = statistics.mean(values_b)
    stdev_b = statistics.stdev(values_b)
    confidence_interval_b = z * stdev_b / sqrt(len(values_b))

    offset = 0.2
    left = x - horizontal_line_width / 2
    top = mean_b - confidence_interval_b
    right = x + horizontal_line_width / 2
    bottom = mean_b + confidence_interval_b

    plt.plot([x + offset, x + offset], [top, bottom], color=color_b)
    plt.plot([left + offset, right + offset], [top, top], color=color_b)
    plt.plot([left + offset, right + offset], [bottom, bottom], color=color_b)

    if legend:
        plt.plot(x + offset, mean_b, 'd', color='blue', label=text[1])
    else:
        plt.plot(x + offset, mean_b, 'd', color='blue')

    plt.legend()


def main():
    mtl_measures = [56.25, 56.25, 56.25, 66.67, 78.57, 71.88, 56.25, 78.12, 53.33, 57.14, 65.62, 56.25, 81.25, 53.33,
                    57.14, 68.75, 53.12, 71.88, 53.33, 71.43, 65.62, 81.25, 56.25, 73.33, 57.14, 71.88, 56.25, 56.25,
                    53.33, 85.71, 56.25, 71.88, 71.88, 73.33, 71.43, 78.12, 56.25, 62.50, 70.00, 57.14, 59.38, 56.25,
                    71.88, 53.33, 82.14, 75.00, 71.88, 62.50, 66.67, 53.57]
    cac_measures = [56.25, 78.12, 59.38, 63.33, 64.29, 68.75, 75.00, 71.88, 56.67, 78.57, 75.00, 71.88, 78.12, 63.33,
                    75.00, 78.12, 59.38, 78.12, 66.67, 71.43, 75.00, 81.25, 59.38, 70.00, 53.57, 78.12, 68.75, 62.50,
                    70.00, 82.14, 71.88, 78.12, 65.62, 63.33, 75.00, 75.00, 59.38, 68.75, 80.00, 57.14, 65.62, 65.62,
                    75.00, 70.00, 78.57, 75.00, 75.00, 78.12, 60.00, 64.29]

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['1-5', '2-5', '3-5', '4-5', '5-5', '6-5', '7-5', '8-5', '9-5', '10-5'],
               rotation=45)
    plt.title('Confidence Interval (95%)')
    acc_mtl = []
    acc_cac = []
    first = True
    cont = 1
    for pos in range(len(mtl_measures)):
        print(pos)
        acc_mtl.append(mtl_measures[pos])
        acc_cac.append(cac_measures[pos])
        if (pos + 1) % 5 == 0:
            if first:
                first = False
                plot_confidence_interval(cont, acc_mtl, acc_cac, legend=True, text=('MTL', 'CAC'))
            plot_confidence_interval(cont, acc_mtl, acc_cac)
            cont += 1
            acc_mtl = []
            acc_cac = []

    # plot_confidence_interval(1, [56.25, 68.75, 68.75, 73.33, 71.43], [56.25, 78.12, 59.38, 63.33, 64.29], legend=True, text=('MTL','CAC'))
    # plot_confidence_interval(2, [68.75, 56.25, 78.12, 53.33, 71.43], [68.75, 75.0, 71.88, 56.67, 78.57])
    # plot_confidence_interval(3, [65.62, 65.62, 56.25, 63.33, 67.86], [75.0, 71.88, 78.12, 60.0, 75.0])
    # plot_confidence_interval(4, [71.88, 46.88, 75.00, 63.33, 64.29], [78.12, 59.38, 78.12, 66.67, 71.43])
    # plot_confidence_interval(5, [56.25, 78.12, 59.38, 53.33, 53.57], [75.0, 81.25, 59.38, 70.0, 53.57])
    # plot_confidence_interval(6, [56.25, 68.75, 68.75, 73.33, 71.43], [56.25, 78.12, 59.38, 63.33, 64.29])
    # plot_confidence_interval(7, [68.75, 56.25, 78.12, 53.33, 71.43], [68.75, 75.0, 71.88, 56.67, 78.57])
    # plot_confidence_interval(8, [65.62, 65.62, 56.25, 63.33, 67.86], [75.0, 71.88, 78.12, 60.0, 75.0])
    # plot_confidence_interval(9, [71.88, 46.88, 75.00, 63.33, 64.29], [78.12, 59.38, 78.12, 66.67, 71.43])
    # plot_confidence_interval(10, [56.25, 78.12, 59.38, 53.33, 53.57], [75.0, 81.25, 59.38, 70.0, 53.57])

    plt.ylabel('CI%')
    plt.xlabel('N-5 folds')
    plt.grid(axis='y')
    plt.show()


if __name__ == '__main__':
    main()
