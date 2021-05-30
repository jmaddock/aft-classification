import json
import yaml
import importlib
import psutil
import argparse
import logging
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.sparse import load_npz

from functools import partial

class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass

METRICS = {
    'accuracy':accuracy_score,
    'precision':precision_score,
    'specificity':partial(precision_score,pos_label = -1),
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

def proba_to_preds(probability_list,threshold=.5):
    preds = []
    for proba in probability_list:
        if proba[1] > threshold:
            preds.append(1)
        else:
            preds.append(-1)
    return preds

def results_to_table(proba,truth):
    result_dict = results_to_json(proba,truth)
    return pd.DataFrame([result_dict])
    #return results

def results_to_json(proba,truth):
    preds = proba_to_preds(proba)
    results_dict = {}
    for metric in METRICS:
        if metric == 'roc_auc':
            results_dict[metric] = METRICS[metric](truth,proba[:, 1])
        else:
            results_dict[metric] = METRICS[metric](truth,preds)
    return results_dict

def train_and_test_split(feature_path, sparse_matrix_path=None):
    with open(feature_path, 'r') as filestream:
        df = pd.DataFrame(json.load(filestream))

    df = generate_sample(df, None)
    df = df.reset_index()

    if sparse_matrix_path:
        with open(sparse_matrix_path, 'rb') as filestream:
            features = load_npz(filestream)

    else:
        features = pd.DataFrame(df['feature_vector'].values.tolist()).to_numpy()

    labels = df['aft_net_sign_helpful'].to_numpy()
    indicies = np.arange(len(labels))

    features_train, features_test, labels_train, labels_test, i_train, i_test = train_test_split(
        features,
        labels,
        indicies,
        test_size=.2,
        stratify=labels,
        shuffle=True,
        random_state=1
    )

    return features_train, features_test, labels_train, labels_test, i_train, i_test

def class_for_name(class_path):
    c = getattr(importlib.import_module(class_path.rsplit('.', 1)[0]), class_path.rsplit('.', 1)[1])()
    return c

def format_model_config(yaml):
    param_grid = []
    for model in yaml:
        clf = class_for_name(yaml[model]['class'])
        params = {}
        for param in yaml[model]['params']:
            key = 'clf__{0}'.format(param)
            params[key] = yaml[model]['params'][param]
        params['clf'] = [clf]
        param_grid.append(params)

    return param_grid

def main():
    parser = argparse.ArgumentParser(description='Convert .csv of AFT data to feature vectors.')
    parser.add_argument('infile',
                        help='a file path to a csv containing vectorized AFT data, or a pickled autosklearn classification object')
    parser.add_argument('outfile',
                        help='a file path to write pickled models and train/test indicies')
    parser.add_argument('-m', '--sparse_matrix',
                        default=None)
    parser.add_argument('-c', '--model_config',
                        help='a file path to a yaml file that contains the parameter grid for gridsearch')
    parser.add_argument('-n', '--sample_size',
                        type=int,
                        default=None)
    parser.add_argument('--cpu_limit',
                        type=int,
                        default=psutil.cpu_count())
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-d', '--debug',
                        type=int)
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

    features_train, features_test, labels_train, labels_test, i_train, i_test = train_and_test_split(args.infile,args.sparse_matrix)

    with open(args.model_config) as filestream:
        model_config = yaml.load(filestream)

    param_grid = format_model_config(model_config)
    scoring = ('roc_auc', 'f1', 'accuracy', 'recall', 'precision')
    pipe = Pipeline([('clf', DummyEstimator())])

    best_models = {}

    for model in param_grid:
        gs = GridSearchCV(
            pipe,
            param_grid,
            scoring=scoring,
            n_jobs=args.cpu_limit,
            pre_dispatch=args.cpu_limit*2,
            refit='roc_auc'
        )
        gs.fit(features_train, labels_train)
        best_models[str(model['clf'][0].__class__.__name__)] = gs

    indices = {
        'train': i_train,
        'test': i_test
    }

    output_dict = {
        'classifiers': best_models,
        'indices': indices
    }

    with open(args.outfile,'wb') as filestream:
        pickle.dump(output_dict,filestream)

if __name__ == "__main__":
    main()