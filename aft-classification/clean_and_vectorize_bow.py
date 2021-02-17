#!/usr/bin/env python
# coding: utf-8

import csv
import json
import argparse
import logging
import functools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

class CleanAndVectorize(object):

    def __init__(self,**kwargs):
        max_df = kwargs.get('max_df',.9)
        max_features = kwargs.get('max_features', 1000)
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            max_df=max_df,
            max_features=max_features
        )
        self.tokenizer = self.vectorizer.build_tokenizer()
        self.cols_to_extract = [
            'aft_id',
            'aft_page',
            'aft_page_revision',
            'aft_user',
            'aft_user_text',
            'aft_comment',
            'aft_noaction',
            'aft_inappropriate',
            'aft_helpful',
            'aft_unhelpful'
        ]

    def process(self, observations, save_tokens=False, remove_zero=True, debug=False):
        if debug:
            observations = observations.sample(debug)
        observations = observations[self.cols_to_extract]
        observations['aft_comment'] = observations['aft_comment'].astype(str)
        observations['aft_net_sign_helpful'] = np.sign(
            observations['aft_helpful'] - observations['aft_unhelpful']).astype(int)
        if remove_zero:
            observations = observations.loc[observations['aft_net_sign_helpful'] != 0]
        if save_tokens:
            observations['tokenized_text'] = observations['aft_comment'].apply(self.tokenizer)
        #observations['feature_vector'] = self.vectorizer.fit_transform(observations['aft_comment'].values).toarray().tolist()
        feature_vectors = self.vectorizer.fit_transform(observations['aft_comment'].values)
        return observations, feature_vectors

def main():
    parser = argparse.ArgumentParser(description='Convert .csv of AFT data to feature vectors.')
    parser.add_argument('infile',
                        help='a file path to a csv containing AFT data')
    parser.add_argument('meta_outfile',
                        help='a file path to meta data for each observation')
    parser.add_argument('sparse_outfile',
                        help='a file path to write a sparse matrix of feature vectors')
    parser.add_argument('--max_features',
                        type=int,
                        default=1000)
    parser.add_argument('--max_df',
                        type=float,
                        default=.9)
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-d', '--debug',
                        type=int)
    parser.add_argument('--save_tokens',
                        action='store_true')
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

    cv = CleanAndVectorize(max_df=args.max_df, max_features=args.max_features)

    dtypes = {
        'aft_id': object,
        'aft_helpful':int,
        'aft_unhelpful':int
    }

    df = pd.read_csv(args.infile, escapechar='\\', encoding='latin-1', dtype=dtypes)
    observations, feature_vectors = cv.process(df, save_tokens=args.save_tokens, debug=args.debug)

    with open(args.meta_outfile,'w') as outfile:
        json.dump(observations.to_dict('records'),outfile)

    with open(args.sparse_outfile, 'wb') as outfile:
        save_npz(outfile, feature_vectors)

if __name__ == "__main__":
    main()




