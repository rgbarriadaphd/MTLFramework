"""
# Author: ruben 
# Date: 26/9/22
# Project: MTLFramework
# File: n_5_fold_cross_validation.py

Description: Generates latex tables with results comparing MTL vs CAC N-Fold cross validation.
"""
import random
from utils.metrics import CrossValidationMeasures

ND = 2


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
            print(f'\\textbf{{{fold}-5}} & {mean} & {stddev} & ${ci}$ \\\\')
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

            print(f'\multirow{{2}}{{*}}{{\\textbf{{{fold}-5}}}} & MTL & {mean_a} & {stddev_a} & ${ci_a}$\\\\')
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
    mtl_measures = [56.25, 56.25, 56.25, 66.67, 78.57, 71.88, 56.25, 78.12, 53.33, 57.14, 65.62, 56.25, 81.25, 53.33,
                    57.14, 68.75, 53.12, 71.88, 53.33, 71.43, 65.62, 81.25, 56.25, 73.33, 57.14, 71.88, 56.25, 56.25,
                    53.33, 85.71, 56.25, 71.88, 71.88, 73.33, 71.43, 78.12, 56.25, 62.50, 70.00, 57.14, 59.38, 56.25,
                    71.88, 53.33, 82.14, 75.00, 71.88, 62.50, 66.67, 53.57]
    cac_measures = [56.25, 78.12, 59.38, 63.33, 64.29, 68.75, 75.00, 71.88, 56.67, 78.57, 75.00, 71.88, 78.12, 63.33,
                    75.00, 78.12, 59.38, 78.12, 66.67, 71.43, 75.00, 81.25, 59.38, 70.00, 53.57, 78.12, 68.75, 62.50,
                    70.00, 82.14, 71.88, 78.12, 65.62, 63.33, 75.00, 75.00, 59.38, 68.75, 80.00, 57.14, 65.62, 65.62,
                    75.00, 70.00, 78.57, 75.00, 75.00, 78.12, 60.00, 64.29]

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

    # ce_measures = [82.24, 85.53, 82.57, 81.97, 81.31]
    ce_accuracy = [71.10, 66.21, 58.45, 68.11, 68.42]
    ce_precision = [75.34, 75.27, 67.19, 77.89, 75.27]
    ce_recall = [83.17, 72.54, 68.25, 73.27, 76.09]
    ce_f1 = [79.06, 73.88, 67.72, 75.51, 75.68]
    global_results(ce_accuracy)
    global_results(ce_precision)
    global_results(ce_recall)
    global_results(ce_f1)
    # print('\subsubsection{By outer fold}')
    # by_outer_fold_results(ce_measures)
