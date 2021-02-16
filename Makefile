all: vectorize train

# shouldn't hardcode input dir
# remove debug flag
vectorize: word2vec/enwiki-20191201-learned_vectors.50_cell.100k.kv
	python ./aft-classification/clean_and_vectorize.py \
		/srv/aft/raw/dump_03-24-20.csv \
		./datasets/vectorized/vectorized_w2v_2021-02-11.json \
		./word2vec/enwiki-20191201-learned_vectors.50_cell.100k.kv \
		--save_tokens

vectorize_bow:
	python ./aft-classification/clean_and_vectorize_bow.py \
		/Users/klogg/research_data/aft/raw/dump_03-24-20.csv \
		./datasets/vectorized/vectorized_bow_100k_2021-02-16.json \
		./datasets/vectorized/vectorized_bow_100k_2021-02-16.npz \
		--max_df .9 \
		--max_features 10 \
		--save_tokens

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


test: ./datasets/vectorized/vectorized_w2v_10k_2021-02-09.json
	python ./aft-classification/train_and_validate.py \
		./datasets/vectorized/vectorized_w2v_10k_2021-02-09.json \
		./model_results/model_results_w2v_10k_2021-02-11.csv \
		-w 4 \
		-n 25 \
		-s ./models/model_10k_2021-02-11.pickle


word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv:
	wget https://analytics.wikimedia.org/datasets/archive/public-datasets/all/ores/topic/vectors/enwiki-20200501-learned_vectors.50_cell.10k.kv -qO- > $@

word2vec/enwiki-20191201-learned_vectors.50_cell.100k.kv:
	wget https://analytics.wikimedia.org/datasets/archive/public-datasets/all/ores/topic/vectors/enwiki-20191201-learned_vectors.50_cell.100k.kv -qO- > $@
