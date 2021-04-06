#!/usr/bin/env python
# coding: utf-8

import json
import logging
import argparse
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from scipy.sparse import load_npz

METRICS = {
    'balanced_accuracy':balanced_accuracy_score,
    'precision':precision_score,
    'recall':recall_score,
    'f1':f1_score,
    'average_precision':average_precision_score,
    'roc_auc':roc_auc_score
}

def generate_sample(df,n,balance=True):
    if balance and n:
        pos = df.loc[df['aft_net_sign_helpful'] > 0].sample(int(n/2))
        neg = df.loc[df['aft_net_sign_helpful'] < 0].sample(int(n/2))
        sample = pos.append(neg)
    elif n:
        sample = df.loc[df['aft_net_sign_helpful'] != 0].sample(n)
    else:
        sample = df.loc[df['aft_net_sign_helpful'] != 0]
    return sample

def results_to_table(preds,truth):
    result_dict = results_to_json(preds,truth)
    return pd.DataFrame([result_dict])
    #return results

def results_to_json(preds,truth):
    results_dict = {metric:METRICS[metric](truth,preds) for metric in METRICS}
    return results_dict

def main():
    parser = argparse.ArgumentParser(description='Convert .csv of AFT data to feature vectors.')
    parser.add_argument('infile',
                        help='a file path to a csv containing vectorized AFT data, or a pickled autosklearn classification object')
    parser.add_argument('outfile',
                        help='a file path to write validation results')
    parser.add_argument('-m', '--sparse_matrix',
                        default=None)
    parser.add_argument('-o', '--output_type',
                        choices=['csv','json'],
                        default='csv')
    parser.add_argument('-n', '--sample_size',
                        type=int,
                        default=None)
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-d', '--debug',
                        type=int)
    parser.add_argument('-s', '--save_model',
                        help='a file path to save a pickled version of the classifiers')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()

    formatter = logging.Formatter(fmt='[%(levelname)s %(asctime)s] %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if args.infile.rsplit('.',1)[-1] == 'json':
        with open(args.infile,'r') as filestream:
            df = pd.DataFrame(json.load(filestream))

        df = generate_sample(df, args.sample_size)

        if args.sparse_matrix:
            with open(args.sparse_matrix,'rb') as filestream:
                features = load_npz(filestream)

        else:
            features = pd.DataFrame(df['feature_vector'].values.tolist()).to_numpy()

        labels = df['aft_net_sign_helpful'].to_numpy()

        features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size = .2,
            stratify = labels,
            shuffle = True,
            random_state = 1
        )

        cls = LogisticRegression(
            random_state = 0,
            max_iter=1000
        )
        cls.fit(
            features_train,
            labels_train,
        )

        preds = cls.predict(features_test)

        if args.output_type == 'json':
            with open(args.outfile, 'w') as filestream:
                results = results_to_json(preds,labels_test)
                json.dump(results, filestream)

        elif args.output_type == 'csv':
            results = results_to_table(preds,labels_test)
            results.to_csv(args.outfile)

if __name__ == "__main__":
    main()