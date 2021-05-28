import json
import argparse
import logging
import re
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

class BadWordCounter(object):

    def __init__(self, regex_list):
        self.pattern = re.compile('({0})'.format(')|('.join(regex_list)))

    def get_bad_word_count(self, sentence):
        return len(self.pattern.findall(sentence))

class CleanAndVectorize(object):

    def __init__(self, en_kvs_path, **kwargs):
        max_df = kwargs.get('max_df',.9)
        max_features = kwargs.get('max_features', 1000)
        self.tfidf_vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            max_df=max_df,
            max_features=max_features
        )
        self.w2v_vectorizer = KeyedVectors.load(en_kvs_path,mmap='r')
        self.tokenizer = self.tfidf_vectorizer.build_tokenizer()
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
            'aft_unhelpful',
            'aft_rating'
        ]

    def get_token_vector(self, token):
         if token in self.w2v_vectorizer:
            return self.w2v_vectorizer[token]
         else:
            return np.zeros(self.w2v_vectorizer.vector_size)

    def get_sentence_vector(self, token_list):
        vector_list = np.array([self.get_token_vector(x) for x in token_list])
        sentence_vector = np.mean(vector_list,axis=0)
        return sentence_vector

    def get_feature_vector(self,observation,add_rating=False):
        feature_vector = self.get_sentence_vector(observation['tokenized_text'])
        if add_rating:
            feature_vector = np.append(feature_vector,observation['aft_rating'])
        feature_vector = feature_vector.tolist()
        return feature_vector

    def process(self, observations, save_tokens=False, remove_zero=True, debug=False, add_rating=False):
        if debug:
            observations = observations.sample(debug)
        observations = observations[self.cols_to_extract]
        observations['aft_comment'] = observations['aft_comment'].astype(str)
        observations['aft_net_sign_helpful'] = np.sign(
            observations['aft_helpful'] - observations['aft_unhelpful']).astype(int)
        if remove_zero:
            observations = observations.loc[observations['aft_net_sign_helpful'] != 0]
        observations['tokenized_text'] = observations['aft_comment'].apply(self.tokenizer)
        observations = observations.loc[observations['tokenized_text'].apply(len) > 0]
        observations['feature_vector'] = observations[['tokenized_text','aft_rating']].apply(
            self.get_feature_vector,
            axis=1,
            add_rating=add_rating)
        if not save_tokens:
            observations.drop(labels='tokenized_text',axis=1,inplace=True)
        return observations

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
    parser.add_argument('--add_rating',
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

    cv = CleanAndVectorize(en_kvs_path=args.embedding_file)

    dtypes = {
        'aft_id': object,
        'aft_helpful':int,
        'aft_unhelpful':int
    }

    df = pd.read_csv(args.infile, escapechar='\\', encoding='latin-1', dtype=dtypes)
    observations = cv.process(df, save_tokens=args.save_tokens, debug=args.debug, add_rating=args.add_rating)

    with open(args.outfile,'w') as outfile:
        json.dump(observations.to_dict('records'),outfile)

if __name__ == "__main__":
    main()