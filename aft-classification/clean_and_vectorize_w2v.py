#!/usr/bin/env python
# coding: utf-8

import csv
import json
import argparse
import logging
import functools
import numpy as np

from revscoring.features import wikitext
from revscoring.datasources.meta import mappers, vectorizers
from revscoring.datasources import revision_oriented
from revscoring.dependencies import solve
from revscoring.features.meta import aggregators

class CleanAndVectorize(object):

    def __init__(self, en_kvs_path):
        self.tokenizer = mappers.lower_case(wikitext.revision.datasources.words)
        self.vectorizer = self.load_vectorizer(en_kvs_path)
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

    def load_vectorizer(self, enwiki_kvs_path):
        enwiki_kvs = vectorizers.word2vec.load_gensim_kv(
            path=enwiki_kvs_path,
            mmap="r"
        )

        vectorize_words = functools.partial(vectorizers.word2vec.vectorize_words, enwiki_kvs)

        revision_text_vectors = vectorizers.word2vec(
            mappers.lower_case(wikitext.revision.datasources.words),
            vectorize_words,
            name="revision.text.en_vectors")

        w2v = aggregators.mean(
            revision_text_vectors,
            vector=True,
            name="revision.text.en_vectors_mean"
        )

        return w2v

    def process(self, filepath, save_tokens=False, debug=False):
        with open(filepath,encoding='latin-1') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', escapechar='\\')
            for i, row in enumerate(csvreader):
                if i == 0:
                    header = row
                elif row[header.index('aft_comment')]:
                    observation = {}
                    for j, cell in enumerate(row):
                        if header[j] in self.cols_to_extract:
                            observation[header[j]] = cell

                    cache = {}

                    cache[revision_oriented.revision.text] = observation['aft_comment']
                    tokenized_text = solve(self.tokenizer, cache=cache, context=None)

                    if save_tokens:
                        observation['tokenized_text'] = tokenized_text

                    cache[self.tokenizer] = tokenized_text
                    observation['feature_vector'] = solve(self.vectorizer, cache=cache, context=None)

                    observation['aft_net_sign_helpful'] = int(np.sign(int(observation['aft_helpful'])-int(observation['aft_unhelpful'])))

                    yield observation

                if debug:
                    if i == debug:
                        break

def main():
    parser = argparse.ArgumentParser(description='Convert .csv of AFT data to feature vectors.')
    parser.add_argument('infile',
                        help='a file path to a csv containing AFT data')
    parser.add_argument('outfile',
                        help='a file path to write feature vectors')
    parser.add_argument('embedding_file',
                        help='a kvs embedding layer file')
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

    cv = CleanAndVectorize(args.embedding_file)

    with open(args.outfile,'w') as outfile:
        observations = [obs for obs in cv.process(args.infile,
                                                  save_tokens=args.save_tokens,
                                                  debug=args.debug)]
        json.dump(observations,outfile)

if __name__ == "__main__":
    main()




