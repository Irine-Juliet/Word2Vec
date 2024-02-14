# training_utils.py
import nltk
import numpy as np
import random
from math import exp
from typing import List, Tuple, Optional
from data_prep import get_corpus, preprocess, get_vocabulary, get_vocab2idx


def load_stopwords() -> set[str]:
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words("english")
    return set(stopwords)

def generate_training_data(corpus: List[str], vocab2idx, window_size: int, k: int) -> List[Tuple[int, List[int], List[int]]]:
    """
    Generates the training data as a list. Each element of the list
    follows the format:

        (center word index, list of context word indicies, list of negative word indices)

    The context word indices are are the indices of the words within the windows size inclusive.
    The negative indicies are sampled from the entire vocabulary,
    excluding the positive and context indices. To keep things simple, we use
    uniform sampling. However, the original word2vec paper uses weighted sampling.
    Example output shown below (note that the numbers are not correct):
    [
        (10, [1, 2, 3], [0, 4, 5, 8, 9]),
        ...
    ]
    """
    result = []
    processed_corpus = preprocess(corpus)
    tokens = [token for sublist in processed_corpus for token in sublist]
    for i, center_word in enumerate(tokens):
        if center_word not in vocab2idx:
            continue
        center_word_idx = vocab2idx[center_word]
        context_indices = []
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
            if j != i and tokens[j] in vocab2idx:
                context_indices.append(vocab2idx[tokens[j]])

        # Generating negative samples
        all_indices = list(range(len(vocab2idx)))
        negative_indices = random.sample([idx for idx in all_indices if idx not in context_indices and idx != center_word_idx], k)

        result.append((center_word_idx, context_indices, negative_indices))

    return result

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the softmax activation function.
    """
    np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    

# Softmax loss and gradients function
def get_softmax_loss_and_gradients(
    v_c: np.ndarray,
    u_idx: int,
    U: np.ndarray,
    negative_samples: Optional[list[int]] = None,
):
    """
    This part implements the softmax loss and returns the gradients with respect to the input.

    Given the center word v_c, the index of the context word u_idx,
    and the word embedding matrix U, compute the softmax loss and
    gradients for both the center word and the context word.

    Args:
      v_c: np.ndarray shape (dim)
      u_idx: int
      U: np.ndarray shape (V, dim)
      negative_samples: Not used (ignore for this part)

    Returns:
      loss: float
      grad_v_c: np.ndarray
      grad_outside_vectors: np.ndarray
    """
    loss, grad_v_c, grad_outside_vectors = None, None, None
    scores = np.dot(U, v_c)

    # compute probabilities
    probs = np.exp(scores) / np.sum(np.exp(scores))

    # Softmax loss: negative log probability
    loss = -np.log(probs[u_idx])

    # Gradient with respect to v_c
    grad_v_c = -U[u_idx] + np.sum(probs[:, np.newaxis] * U, axis=0)

    # Gradient with respect to U
    grad_outside_vectors = probs[:, np.newaxis] * v_c
    grad_outside_vectors[u_idx] -= v_c
    # END OF YOUR CODE

    return loss, grad_v_c, grad_outside_vectors

# negative sampling loss and corresponding gradients
def get_negative_sampling_loss_and_gradients(
    v_c: np.ndarray,
    u_idx: int,
    U: np.ndarray,
    negative_samples_idx: list[int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    This part implements the negative sampling loss and also returns the gradients.

    Given the center word v_c, the index of the context word u_idx,
    and the word embedding matrix U, compute the negative sampling loss and
    gradients for both the center word and the context word.

    Args:
        v_c: np.ndarray shape (dim)
        u_idx: int
        v_c_idx: int, the index of the center word that we are considering
          used to eliminate this from being selected as negative
        U: np.ndarray shape (V, dim)
        dataset: list[tuple[int, int]]
        k: int, number of negative samples

    Returns:
        loss: float
        grad_v_c: np.ndarray
        grad_outside_vector: np.ndarray
    """

    loss, grad_v_c, grad_outside_vectors = None, None, None
    true_outside_vector = U[u_idx]
    true_score = np.dot(true_outside_vector, v_c)
    true_loss = -np.log(sigmoid(true_score))

    # Initialize grads
    grad_v_c = np.zeros_like(v_c)
    grad_outside_vectors = np.zeros_like(U)

    grad_v_c += (sigmoid(true_score) - 1) * true_outside_vector
    grad_outside_vectors[u_idx] += (sigmoid(true_score) - 1) * v_c

    # Initialize total loss
    loss = true_loss

    for neg_idx in negative_samples_idx:
        neg_vector = U[neg_idx]
        neg_score = np.dot(neg_vector, v_c)
        loss -= np.log(sigmoid(-neg_score))

        # Gradient updates
        grad_v_c += (sigmoid(-neg_score) - 1) * (-neg_vector)
        grad_outside_vectors[neg_idx] += (sigmoid(-neg_score) - 1) * (-v_c)

    return loss, grad_v_c, grad_outside_vectors

def get_nearest_neighbors(word: str, embeddings: np.ndarray, vocab2idx: dict[str, int], idx2vocab: list[str], k: int = 5) -> list[tuple[str, float]]:
    """
    Get the k nearest neighbors for a given word.
    """
    idx = vocab2idx[word]
    embedding = embeddings[idx]
    distances = np.linalg.norm(embedding - embeddings, axis=1)
    sorted_distances = np.argsort(distances)
    return [(idx2vocab[idx], distances[idx]) for idx in sorted_distances[:k]]

def print_nearest_neighbors(model: Word2Vec, k: int = 5, num_print: int = 20) -> None:
    # print nearest neighbors
    j = 0
    for word in model.vocab:
        j += 1
        if word in stopwords:
            continue
        print(
            word,
            get_nearest_neighbors(
                word,
                model.get_embeddings_avg(),
                model.vocab2idx,
                model.idx2vocab,
            ),
        )
        if j > num_print:
            break
    print(
        word,
        get_nearest_neighbors(
            word, model.get_embeddings_avg(), model.vocab2idx, model.idx2vocab, k=k
        ),
    )


