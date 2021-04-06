all: vectorize train

# shouldn't hardcode input dir
# remove debug flag
vectorize: word2vec/aft_2021-03-30_learned_vectors.50_cell.10k.kv
	python ./aft-classification/clean_and_vectorize_w2v_old.py \
		/Users/klogg/research_data/aft/raw/dump_03-24-20.csv \
		./datasets/vectorized/vectorized_w2v_2021-03-30.json \
		./word2vec/aft_2021-03-30_learned_vectors.50_cell.10k.kv \
		--save_tokens

vectorize_w2v_w_rating: word2vec/aft_2021-03-30_learned_vectors.50_cell.10k.kv
	python ./aft-classification/clean_and_vectorize_w2v.py \
		/Users/klogg/research_data/aft/raw/dump_03-24-20.csv \
		./datasets/vectorized/vectorized_w2v_rating_2021-03-31.json \
		./word2vec/aft_2021-03-30_learned_vectors.50_cell.10k.kv \
		--save_tokens

vectorize_bow:
	python ./aft-classification/clean_and_vectorize_bow.py \
		/Users/klogg/research_data/aft/raw/dump_03-24-20.csv \
		./datasets/vectorized/vectorized_bow_100k_2021-02-16.json \
		./datasets/vectorized/vectorized_bow_100k_2021-02-16.npz \
		--max_df .9 \
		--max_features 100000 \
		--save_tokens

vectorize_bow_w_rating:
	python ./aft-classification/clean_and_vectorize_bow.py \
		/Users/klogg/research_data/aft/raw/dump_03-24-20.csv \
		./datasets/vectorized/vectorized_bow_1k_rating_2021-03-30.json \
		./datasets/vectorized/vectorized_bow_1k_rating_2021-03-30.npz \
		--max_df .9 \
		--max_features 1000 \
		--save_tokens \
		--add_rating

train_w2v: ./datasets/vectorized/vectorized_w2v_100k_2021-02-11.json
	python ./aft-classification/train_and_validate.py \
		./datasets/vectorized/vectorized_w2v_100k_2021-02-11.json \
		./model_results/model_results_w2v_100k_2021-02-11.csv \
		-w 4 \
		-s ./models/model_w2v_100k_2021-02-11.pickle

train_bow: ./datasets/vectorized/vectorized_bow_100k_2021-02-16.json \
	./datasets/vectorized/vectorized_bow_100k_2021-02-16.npz
	python ./aft-classification/train_and_validate.py \
		./datasets/vectorized/vectorized_bow_100k_2021-02-16.json \
		./model_results/model_results_bow_100k_2021-02-16.csv \
		-m ./datasets/vectorized/vectorized_bow_100k_2021-02-16.npz \
		-w 4 \
		-s ./models/model_bow_100k_2021-02-16.pickle

train_logistic_bow_w_rating: ./datasets/vectorized/vectorized_bow_1k_rating_2021-03-30.json \
	./datasets/vectorized/vectorized_bow_1k_rating_2021-03-30.npz
	python aft-classification/train_and_validate_logistic_regression.py \
		datasets/vectorized/vectorized_bow_1k_rating_2021-03-30.json \
		./model_results/model_results_logistic_bow_1k_rating_2021-03-30.csv \
		-m datasets/vectorized/vectorized_bow_1k_rating_2021-03-30.npz

train_logistic_bow: ./datasets/vectorized/vectorized_bow_100k_2021-02-16.json \
	./datasets/vectorized/vectorized_bow_100k_2021-02-16.npz
	python aft-classification/train_and_validate_logistic_regression.py \
		datasets/vectorized/vectorized_bow_100k_2021-02-16.json \
		./model_results/model_results_logistic_bow_100k_2021-03-11.csv \
		-m datasets/vectorized/vectorized_bow_100k_2021-02-16.npz

train_logistic_w2v: datasets/vectorized/vectorized_w2v_100k_2021-02-11.json
	python aft-classification/train_and_validate_logistic_regression.py \
		datasets/vectorized/vectorized_w2v_100k_2021-02-11.json \
		./model_results/model_results_logistic_w2v_100k_2021-03-11.csv

train_w2v_w_rating: ./datasets/vectorized/vectorized_w2v_rating_2021-03-31.json
	python ./aft-classification/train_and_validate.py \
		./datasets/vectorized/vectorized_w2v_rating_2021-03-31.json \
		./model_results/model_results_w2v_rating_2021-04-16.csv \
		-s ./models/model_w2v_rating_2021-04-16.pickle \
		--time_limit 43200

## download and/or create embedding files
./word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv:
	wget https://analytics.wikimedia.org/datasets/archive/public-datasets/all/ores/topic/vectors/enwiki-20200501-learned_vectors.50_cell.10k.kv -qO- > $@

./word2vec/enwiki-20191201-learned_vectors.50_cell.100k.kv:
	wget https://analytics.wikimedia.org/datasets/archive/public-datasets/all/ores/topic/vectors/enwiki-20191201-learned_vectors.50_cell.100k.kv -qO- > $@

./word2vec/aft_2021-03-30_learned_vectors.50_cell.10k.kv: ./datasets/raw/dump_03-24-20.csv
	python ./aft_classification/train_w2v_embedding \
		./datasets/raw/dump_03-24-20.csv \
		./word2vec/aft_2021-03-30_learned_vectors.50_cell.10k.kv \
		--max_features 100
		--max_vocab 10000

## download aft dataset
./datasets/raw/dump_03-24-20.csv:
 	wget https://ndownloader.figshare.com/files/3256943 -qO- > $@