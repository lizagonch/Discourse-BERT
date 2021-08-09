from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

import argparse
import zipfile
import os
import time
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def mesure_performance(file_name):
    names = ['segment', 'pred_class', 'real_class']
    data = pd.read_csv(file_name, delimiter='\t', header=None, names=names)
    mean_data = data.groupby(['segment']).mean()
    mean_data['pred_class_1'] = mean_data.pred_class.apply(lambda x: round(x + 0.01))
    mean_data.real_class = mean_data.real_class.apply(int)

    y_true = mean_data.real_class.values
    y_pred = mean_data.pred_class_1.values

    av_prec = average_precision = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print('Precision = {}'.format(round(prec, 2)))
    print('Recall = {}'.format(round(recall, 2)))
    print('F1 score = {}'.format(round(f1, 2)))
    print('Average precision = {}'.format(round(av_prec, 2)))
    return av_prec, prec, recall, f1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", default="epoch.tsv", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    args = parser.parse_args()

    mesure_performance(args.results_dir)

if __name__ == "__main__":
    main()