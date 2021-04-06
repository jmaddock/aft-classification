#!/usr/bin/env python
# coding: utf-8

import json
import logging
import argparse
import pickle
import pandas as pd
import numpy as np
import psutil

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from autosklearn.metrics import balanced_accuracy, precision, recall, f1, roc_auc, average_precision
import autosklearn.classification
from scipy.sparse import load_npz

METRICS = [
    balanced_accuracy,
    precision,
    recall,
    f1,
    average_precision,
    roc_auc
]

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

def results_to_table(cls):
    results = pd.DataFrame.from_dict(cls.cv_results_)
    params = results['params'].apply(pd.Series)
    results = pd.concat([results.drop(['params'],axis=1),params],axis=1)
    return results

def results_to_json(cls):
    results = pd.DataFrame.from_dict(cls.cv_results_).to_dict('records')
    return results

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
    parser.add_argument('-k',
                        type=int,
                        help='number of cross validation folds')
    parser.add_argument('-n', '--sample_size',
                        type=int,
                        default=None)
    parser.add_argument('--cpu_limit',
                        type=int,
                        default=psutil.cpu_count())
    parser.add_argument('--memory_limit',
                        type=int,
                        default=(psutil.virtual_memory()[1] >> 20) * .75)
    parser.add_argument('--time_limit',
                        type=int,
                        default=43200)
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
        feat_type = ['Numerical'] * np.shape(features)[1]

        features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size = .2,
            stratify = labels,
            shuffle = True,
            random_state = 1
        )

        cls = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=args.time_limit,
            #per_run_time_limit=30,
            include_preprocessors=['no_preprocessing'],
            ensemble_size=0,
            scoring_functions=METRICS,
            #resampling_strategy = StratifiedKFold,
            #resampling_strategy_arguments={'folds': args.k},
            n_jobs = args.cpu_limit,
            memory_limit = args.memory_limit
        )
        cls.fit(
            features_train,
            labels_train,
            features_test,
            labels_test,
            feat_type=feat_type
        )

    else:
        with open(args.infile, 'rb') as filestream:
            cls = pickle.load(filestream)

    if args.save_model:
        with open(args.save_model,'wb') as filestream:
            pickle.dump(cls,filestream)

    if args.output_type == 'json':
        with open(args.outfile,'w') as filestream:
            results = results_to_json(cls)
            json.dump(results,filestream)

    elif args.output_type == 'csv':
        results = results_to_table(cls)
        results.to_csv(args.outfile)


if __name__ == "__main__":
    main()