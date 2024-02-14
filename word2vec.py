import os
import pathlib
import pickle
import numpy as np
from typing import Optional
from data_preparation import get_vocabulary, get_vocab2idx
from training_utils import generate_training_data, get_softmax_loss_and_gradients, get_negative_sampling_loss_and_gradients,load_stopwords, print_nearest_neighbors
from word2vec_model import Word2Vec  

class Word2Vec:
    def __init__(
        self,
        corpus: list[str],
        embedding_dim: int = 20,
        learning_rate: float = 0.01,
        num_negative_samples: int = 10,
        window_size: int = 5,
        loss_method: str = "softmax",  # or "negative_sampling"
        save_freq: int = 1000,
        cache_path: str = "data/",
        log_freq: int = 1000,
        vocab_size: int = 2000,
    ) -> None:
        """
        Initializes the model parameters and learning hyperparameters.

        Args:
            corpus: list[str], list of documents
            embedding_dim: int, dimension of the word embedding
            learning_rate: float, learning rate for SGD
            num_negative_samples: int, number of negative samples to use for negative sampling
            window_size: int, window size for context words
            loss_method: str, "softmax" or "negative_sampling"
            save_freq: int, how often to save the model
            cache_path: str, where to save the model
            log_freq: int, how often to print the progress
            vocab_size: int, size of the vocabulary
        """
        self.corpus = corpus
        self.vocab = get_vocabulary(self.corpus, vocab_size=vocab_size)
        self.vocab2idx = get_vocab2idx(self.vocab)
        self.idx2vocab = {v: k for k, v in self.vocab2idx.items()}
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_negative_samples = num_negative_samples
        self.window_size = window_size
        self.log_freq = log_freq
        self.save_freq = save_freq

        # the main parameters of the model that we are going to train
        self.center_vecs = (np.random.rand(len(self.vocab), embedding_dim) - 0.5) / embedding_dim
        self.outside_vecs = np.zeros((len(self.vocab), embedding_dim))

        assert loss_method in ["softmax", "negative_sampling"]
        if loss_method == "softmax":
            self.loss_and_grad_fn = get_softmax_loss_and_gradients
        else:
            self.loss_and_grad_fn = get_negative_sampling_loss_and_gradients

        # cache the training data
        self.cache_path = cache_path
        data_cache_path = f"{cache_path}/data.npy"
        if not os.path.exists(data_cache_path):
            pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
            self.data = generate_training_data(
                corpus, self.vocab2idx, window_size, num_negative_samples
            )
            # save mmap
            with open(data_cache_path, "wb") as f:
                pickle.dump(self.data, f)
        else:
            # load from mmap
            with open(data_cache_path, "rb") as f:
                self.data = pickle.load(f)

    def train_step(
        self,
        center_word_idx: int,
        outside_words_indexes: list[int],
        negative_idxs: Optional[list[int]] = None,  # only used for negative sampling
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Run the train step for a given center word, outside (context) words, and negative samples (if applicable)"""

        loss = 0.0
        grad_center_vectors = np.zeros(self.center_vecs.shape)
        gradoutside_vectors = np.zeros(self.outside_vecs.shape)

        for ow_idx in outside_words_indexes:
            center_word_idx = center_word_idx
            loss_j, grad_v_c, grad_outside_vector_j = self.loss_and_grad_fn(
                self.center_vecs[center_word_idx],
                ow_idx,
                self.outside_vecs,
                negative_idxs,
            )
            loss += loss_j
            grad_center_vectors[center_word_idx] += grad_v_c
            gradoutside_vectors += grad_outside_vector_j

        return loss, grad_center_vectors, gradoutside_vectors

    def save(self, current_step: int) -> None:
        np.save(f"{self.cache_path}/center_vecs_{current_step}.npy", self.center_vecs)
        np.save(f"{self.cache_path}/outside_vecs_{current_step}.npy", self.outside_vecs)

    def load(self, checkpoint_path: str, step: int) -> None:
        # load the latest checkpoint
        self.center_vecs = np.load(f"{checkpoint_path}/center_vecs_{step}.npy")
        self.outside_vecs = np.load(f"{checkpoint_path}/outside_vecs_{step}.npy")

    def get_embeddings_avg(self) -> np.ndarray:
        return (self.center_vecs + self.outside_vecs) / 2

    def get_embeddigns_concat(self) -> np.ndarray:
        return np.concatenate((self.center_vecs, self.outside_vecs), axis=1)

    def train(
        self, batch_size: int = 32, num_epochs: int = 10, num_steps: Optional[int] = None
    ) -> None:
        """
        The training loop of the model
        Batch size is simulated by gradient accumulation (code doesn't support batch dimension)

        """
        gradient_accumulation_steps = batch_size

        global_steps = 0
        steps = 0
        loss = 0.0
        grad_center = np.zeros(self.center_vecs.shape)
        grad_outside = np.zeros(self.outside_vecs.shape)
        total_batches = len(self.data) // batch_size
        if num_steps is not None:
            stop_at = num_steps
            num_epochs = 10000  # not used anymore
        else:
            stop_at = total_batches * num_epochs
        done = False

        for epoch in range(num_epochs):
            if done:
                break
            local_step = 0
            for center_idx, outside_word_indexes, negative_idxs in self.data:
                current_loss, center_grad, outside_grad = self.train_step(
                    center_idx,
                    outside_word_indexes,
                    negative_idxs,
                )
                if steps % gradient_accumulation_steps == 0:
                    grad_center += center_grad
                    grad_center /= batch_size
                    self.center_vecs -= self.learning_rate * grad_center

                    grad_outside += outside_grad
                    grad_outside /= batch_size
                    self.outside_vecs -= self.learning_rate * grad_outside

                    # zero out the gradients
                    grad_center = np.zeros(self.center_vecs.shape)
                    grad_outside = np.zeros(self.outside_vecs.shape)
                    global_steps += 1

                    if global_steps % self.log_freq == 0:
                        progress_percent = round(global_steps / stop_at * 100, 2)
                        print(
                            f"ep {epoch} step {local_step} global step {global_steps} n_batches {total_batches} progress {progress_percent}%: loss {(loss / steps):.3f}"
                        )

                    if global_steps % self.save_freq == 0:
                        self.save(global_steps)

                loss += current_loss
                steps += 1
                local_step += 1

                if global_steps >= stop_at:
                    done = True
                    break