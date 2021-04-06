import argparse
import psutil
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def tokenize(observations):
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
    )
    tokenizer = vectorizer.build_tokenizer()
    observations['aft_comment'] = observations['aft_comment'].astype(str)
    tokenized_text = observations['aft_comment'].apply(tokenizer).values
    return tokenized_text

def main():
    parser = argparse.ArgumentParser(description='Convert .csv of AFT data to feature vectors.')
    parser.add_argument('infile',
                        help='a file path to a csv containing AFT data')
    parser.add_argument('outfile',
                        help='a file path to meta data for each observation')
    parser.add_argument('--max_features',
                        type=int,
                        default=100)
    parser.add_argument('--max_vocab',
                        type=int,
                        default=10000)
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-d', '--debug',
                        type=int)
    parser.add_argument('-s', '--save_model',
                        help='a file path to save a pickled version of the classifiers')
    parser.add_argument('--cpu_limit',
                        type=int,
                        default=psutil.cpu_count())
    args = parser.parse_args()

    dtypes = {
        'aft_id': object,
        'aft_helpful': int,
        'aft_unhelpful': int
    }
    df = pd.read_csv(args.infile, escapechar='\\', encoding='latin-1', dtype=dtypes)
    tokenized_text = tokenize(df)
    model = Word2Vec(sentences=tokenized_text,
                     vector_size=args.max_features,
                     window=5,
                     workers=args.cpu_limit,
                     sg=1,
                     max_final_vocab=args.max_vocab)
    if args.save_model:
        model.save(args.save_model)
    word_vectors = model.wv
    word_vectors.save(args.outfile)