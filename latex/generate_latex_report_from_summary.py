"""
# Author: ruben 
# Date: 14/10/22
# Project: MTLFramework
# File: generate_latex_report_from_summary.py

Description: "Enter feature description here"
"""
import os.path
import sys
import pprint

from utils.metrics import CrossValidationMeasures

AVAILABLE_SECTIONS = ['configuration', 'global_performance', 'performance_by_outer_folds']
ND = 2


class SummaryParser:
    """
    Parse experiment summary into a latex format
    """

    def __init__(self, summary_path, output_path, sections=None):
        """
        Class constructor
        """
        self._summary_path = summary_path
        self._output_path = output_path
        self._sections = sections
        self._latex_lines = []
        self._summary_data = {'configuration': {},
                              'fold_data': {
                                  'accuracy': [],
                                  'precision': [],
                                  'recall': [],
                                  'f1': [],
                                  'conf_matrix': []
                              }}
        self._cvm_acc = None
        self._cvm_pre = None
        self._cvm_rec = None
        self._cvm_f1 = None
        self._n_measurements = -1

    def _check_global_item(self, line):
        if 'Architecture' in line:
            self._summary_data['configuration']['Architecture'] = line.split(':')[1].strip()
        if 'Normalized' in line:
            self._summary_data['configuration']['Normalized'] = line.split(':')[1].strip()
        if 'Save model' in line:
            self._summary_data['configuration']['Save model'] = line.split(':')[1].strip()
        if 'Plot Loss' in line:
            self._summary_data['configuration']['Plot Loss'] = line.split(':')[1].strip()
        if 'Device' in line:
            self._summary_data['configuration']['Device'] = line.split(':')[1].strip()
        if 'Cross validation' in line:
            self._summary_data['configuration']['Cross validation'] = line.split(':')[1].strip()
        if 'Require Grad:' in line:
            self._summary_data['configuration']['Require Grad'] = line.split(':')[1].strip()
        if 'Weights Init' in line:
            self._summary_data['configuration']['Weights Init'] = line.split(':')[1].strip()
        if 'Save accuracy' in line:
            self._summary_data['configuration']['Save accuracy'] = line.split(':')[1].strip()
        if 'Control train/test' in line:
            self._summary_data['configuration']['Control train/test'] = line.split(':')[1].strip()
        if 'Plot ROC' in line:
            self._summary_data['configuration']['Plot ROC'] = line.split(':')[1].strip()
        if 'Dynamic freeze' in line:
            self._summary_data['configuration']['Dynamic freeze'] = line.split(':')[1].strip()
        if 'Epochs' in line:
            self._summary_data['configuration']['Epochs'] = line.split(':')[1].strip()
        if 'Batch size' in line:
            self._summary_data['configuration']['Batch size'] = line.split(':')[1].strip()
        if 'Learning rate' in line:
            self._summary_data['configuration']['Learning rate'] = line.split(':')[1].strip()
        if 'Weight Decay' in line:
            self._summary_data['configuration']['Weight Decay'] = line.split(':')[1].strip()
        if 'Criterion' in line:
            self._summary_data['configuration']['Criterion'] = line.split(':')[1].strip()
        if 'Optimizer' in line:
            self._summary_data['configuration']['Optimizer'] = line.split(':')[1].strip()

    def _check_fold_item(self, line, next=None):
        if 'Accuracy' in line:
            self._summary_data['fold_data']['accuracy'].append(float(line.split(':')[1].strip()))
        if 'Precision' in line:
            self._summary_data['fold_data']['precision'].append(float(line.split(':')[1].strip()))
        if 'Recall' in line:
            self._summary_data['fold_data']['recall'].append(float(line.split(':')[1].strip()))
        if 'F1' in line:
            self._summary_data['fold_data']['f1'].append(float(line.split(':')[1].strip()))
        if 'Confusion Matrix' in line:
            cm = []
            upper_values = line.split(':')[1].split(',')
            cm.append([int(upper_values[0].strip()), int(upper_values[1].strip())])
            lower_values = next.split(',')
            cm.append([int(lower_values[0].strip()), int(lower_values[1].strip())])
            self._summary_data['fold_data']['conf_matrix'].append(cm)

    def _parse_summary_file(self):
        """
        Parses summary file to dict
        """
        assert os.path.exists(self._summary_path)
        with open(self._summary_path, 'r') as f:
            lines = f.readlines()
            for pos, line in enumerate(lines):
                self._check_global_item(line)
                next_line = lines[pos + 1] if 'Confusion Matrix' in line else None
                self._check_fold_item(line, next=next_line)

        self._cvm_acc = CrossValidationMeasures(self._summary_data['fold_data']['accuracy'], percent=True,
                                                formatted=True)
        self._cvm_pre = CrossValidationMeasures(self._summary_data['fold_data']['precision'], percent=True,
                                                formatted=True)
        self._cvm_rec = CrossValidationMeasures(self._summary_data['fold_data']['recall'], percent=True, formatted=True)
        self._cvm_f1 = CrossValidationMeasures(self._summary_data['fold_data']['f1'], percent=True, formatted=True)

        assert len(self._summary_data['fold_data']['precision']) == len(
            self._summary_data['fold_data']['recall']) == len(self._summary_data['fold_data']['accuracy']) == len(
            self._summary_data['fold_data']['f1'])
        self._n_measurements = len(self._summary_data['fold_data']['accuracy'])

    def _generate_configuration(self):
        """
        Creates global configuration parameter list
        """
        aux = ['\subsection{Experiment Configuration}\n']

        for key, value in self._summary_data['configuration'].items():
            aux.append(f'\\noindent {key} : {value} \\\\')

        aux.append('\n\n')
        self._latex_lines.append(aux)

    def _generate_global_performance(self):
        """
        Creates global configuration parameter list
        """

        aux = ['\subsection{Global Performance}\n']

        accuracy = self._summary_data['fold_data']['accuracy']

        aux.append('\\begin{figure}[H]')
        aux.append('\centering')
        aux.append("$Acc=\\begin{pmatrix}")
        sacc = ''
        for pos, elem in enumerate(accuracy):

            if (pos + 1) % 5 == 0:
                sacc += f'{elem:.{ND}f} \\\\'
                aux.append(sacc)
                sacc = ''
            else:
                sacc += f'{elem:.{ND}f} & '

        aux.append("\end{pmatrix}$")
        aux.append('\caption{Accuracy of the model on each fold}')
        aux.append('\end{figure}\n\n')

        aux.append("\\begin{table}[H]")
        aux.append("\centering")
        aux.append("\\begin{tabular}{|c|c|c|c|c|}")
        aux.append("\hline")
        aux.append("&\\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall}  & \\textbf{F1-score}\\\\")
        aux.append("\hline")
        aux.append("\hline")
        aux.append(
            f'\\textbf{{Mean}} & {self._cvm_acc.mean()} & {self._cvm_pre.mean()} & {self._cvm_rec.mean()}  & {self._cvm_f1.mean()}\\\\')
        aux.append("\hline")
        aux.append(
            f'\\textbf{{Std Dev}} & {self._cvm_acc.stddev()} & {self._cvm_pre.stddev()} & {self._cvm_rec.stddev()}  & {self._cvm_f1.stddev()}\\\\')
        aux.append("\hline")
        aux.append(
            f'\\textbf{{CI(95\%)}} & ${self._cvm_acc.interval()}$ & ${self._cvm_pre.interval()}$ & ${self._cvm_rec.interval()}$  & ${self._cvm_f1.interval()}$\\\\')
        aux.append("\hline")
        aux.append("\end{tabular}")
        aux.append('\caption{Global model performance based on accuracy, precision, recall and F1-score metrics}')
        aux.append("\end{table}")

        aux.append('\n\n')
        self._latex_lines.append(aux)

    def _generate_fold_performance(self):
        """
        Creates global configuration parameter list
        """
        aux = ['\subsection{Outer folds performance comparison}\n']

        aux.append('\\begin{table}[H]')
        aux.append('\centering')
        aux.append('\\begin{tabular}{|c||c|c|c||c|c|c|}')
        aux.append('\hline')
        aux.append('& \multicolumn{3}{c||}{\\textbf{Accuracy}} & \multicolumn{3}{c|}{\\textbf{Precision}} \\\\')
        aux.append('\hline')
        aux.append('\hline')
        aux.append('\\textbf{Fold ID} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{CI(95\%)} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{CI(95\%)}\\\\')
        aux.append('\hline')

        accuracy = self._summary_data['fold_data']['accuracy']
        precision = self._summary_data['fold_data']['precision']
        acc = []
        pre = []
        fold = 0
        for i in range(self._n_measurements):
            if (i + 1) % 5 == 0:
                fold += 1
                acc.append(accuracy[i])
                pre.append(precision[i])
                cvm_acc = CrossValidationMeasures(acc, percent=True, formatted=True)
                cvm_pre = CrossValidationMeasures(pre, percent=True, formatted=True)
                aux.append(f'\\textbf{{{fold}-5}} & \cellcolor{{blue!25}}{cvm_acc.mean()} & {cvm_acc.stddev()} & ${cvm_acc.interval()}$ & {cvm_pre.mean()} & {cvm_pre.stddev()} & ${cvm_pre.interval()}$ \\\\')
                aux.append('\hline')
                acc = []
                pre = []
            else:
                acc.append(accuracy[i])
                pre.append(precision[i])

        aux.append('\hline')
        aux.append('& \multicolumn{3}{c||}{\\textbf{Recall}} & \multicolumn{3}{c|}{\\textbf{F1-score}} \\\\')
        aux.append('\hline')
        aux.append('\hline')
        aux.append('\\textbf{Fold ID} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{CI(95\%)} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{CI(95\%)}\\\\')
        aux.append('\hline')

        recall = self._summary_data['fold_data']['recall']
        f1 = self._summary_data['fold_data']['f1']
        rec = []
        f1 = []
        fold = 0
        for i in range(self._n_measurements):
            if (i + 1) % 5 == 0:
                fold += 1
                rec.append(accuracy[i])
                f1.append(precision[i])
                cvm_rec = CrossValidationMeasures(rec, percent=True, formatted=True)
                cvm_f1 = CrossValidationMeasures(f1, percent=True, formatted=True)
                aux.append(
                    f'\\textbf{{{fold}-5}} & {cvm_rec.mean()} & {cvm_rec.stddev()} & ${cvm_rec.interval()}$ & {cvm_f1.mean()} & {cvm_f1.stddev()} & ${cvm_f1.interval()}$ \\\\')
                aux.append('\hline')
                rec = []
                f1 = []
            else:
                rec.append(accuracy[i])
                f1.append(precision[i])
        aux.append('\end{tabular}')
        aux.append('\caption{Outer fold model performance based on accuracy, precision, recall and F1-score metrics}')
        aux.append('\end{table}')
        aux.append('\n\n')
        self._latex_lines.append(aux)

    def _save_file(self):
        """
        Write file with latex format
        """
        with open(self._output_path, 'w') as f:
            f.write('\n'.join(self._latex_lines) + '\n')

    def generate(self):
        """
        Create all latex items
        """
        self._parse_summary_file()

        for section in self._sections:
            assert section in AVAILABLE_SECTIONS
            if section == 'configuration':
                self._generate_configuration()
            elif section == 'global_performance':
                self._generate_global_performance()
            elif section == 'performance_by_outer_folds':
                self._generate_fold_performance()

        # flatten list of seubsections
        self._latex_lines = [item for sublist in self._latex_lines for item in sublist]
        self._save_file()


if __name__ == '__main__':
    args = sys.argv
    summary_file = '/home/ruben/PycharmProjects/MTLFramework/output/train/train_20221010_141912/summary.out'
    output_file = '/home/ruben/PycharmProjects/MTLFramework/output/train/train_20221010_141912/summary.ltx'
    section_list = ['configuration', 'global_performance', 'performance_by_outer_folds']
    sp = SummaryParser(summary_path=summary_file, output_path=output_file, sections=section_list)
    sp.generate()
