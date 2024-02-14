# main.py
# install
!pip install datasets

# imports
import os
import random
import re
import numpy as np
from data_prep import get_corpus, preprocess, get_vocabulary, get_vocab2idx
from training_utils import generate_training_data, get_softmax_loss_and_gradients, get_negative_sampling_loss_and_gradients,load_stopwords, print_nearest_neighbors
from word2vec import Word2Vec  


# some initializations
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

# Hyperparameters setup 
embedding_dim = 20
learning_rate = 0.7
training_method = "softmax"
num_negative_samples = 10
window_size = 2
batch_size = 64
num_epochs = 1
save_freq = 1000000
cache_path = "data"
vocab_size = 5000
limit_data_size = 100

corpus = get_corpus(size=limit_data_size)

def get_model(training_method: str = "softmax") -> Word2Vec:
    word2vec = Word2Vec(
        corpus,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        loss_method=training_method,
        num_negative_samples=num_negative_samples,
        window_size=window_size,
        save_freq=save_freq,
        vocab_size=vocab_size,
        cache_path=cache_path,
    )
    word2vec.train(batch_size=batch_size, num_epochs=num_epochs)
    return word2vec

def main():
    word2vec_softmax = get_model(training_method="softmax")
    word2vec_negative_sampling = get_model(training_method="negative_sampling")
    word2vec_softmax.save("final-softmax")
    word2vec_negative_sampling.save("final-negative_sampling")

    stopwords = load_stopwords()

    print("Nearest neighbors for softmax model")
    print_nearest_neighbors(word2vec_softmax)
    print("\n---\nNearest neighbors for negative sampling model")
    print_nearest_neighbors(word2vec_negative_sampling)

if __name__ == "__main__":
    main()
