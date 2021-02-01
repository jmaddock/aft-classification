#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
import autosklearn.classification

def generate_sample(df,n):
    pos = df.loc[df['aft_net_sign_helpful'] > 0].sample(n)
    neg = df.loc[df['aft_net_sign_helpful'] < 0].sample(n)
    sample = pos.append(neg)
    return sample

def k_fold_split(n):
    pass

# can probably add resampling stratagy param and skip folds code
def train_and_validate(features_train, features_test, lables_train, lables_test):
    pass

infile_path = '/Users/klogg/dev/aft-classification/datasets/vectorized/aft_vectorized_01-29-21.json'
sample_size = 25
k = 5
threads = 4

def main():
    with open(infile_path,'r') as filestream:
        full_df = pd.DataFrame(json.load(filestream))
    sample_df = generate_sample(full_df,sample_size)
        
    features = pd.DataFrame(sample_df['feature_vector'].values.tolist()).to_numpy()
    labels = sample_df['aft_net_sign_helpful'].to_numpy()

    skf = StratifiedKFold(n_splits=k,shuffle=True)
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        ensemble_size=0,
        scoring_functions=[balanced_accuracy, precision, recall, f1],
        resampling_strategy = StratifiedKFold,
        n_jobs = threads
    )
    cls.fit(features, labels)

if __name__ == "__main__":
    main()