import re
from collections import Counter
from datasets import load_dataset
from typing import List
from tqdm import tqdm

# data loading. uses a small subset of OpenWebText
def get_corpus(size: int = 1000) -> list[str]:
    dataset = load_dataset("stas/openwebtext-10k")
    corpus = dataset["train"]["text"][:size]
    return corpus


# a tiny corpus for testing purposes
def get_tiny_corpus() -> list[str]:
    document1 = "Natural language processing is a subfield of computer science, linguistics, and machine learning."
    document2 = "It is concerned with giving computers the ability to support and manipulate natural language."
    document3 = "It involves processing natural language datasets using rule-based or probabilistic machine learning approaches."
    document4 = "The goal is a computer capable of understanding the contents of documents through machine learning."
    corpus = [document1, document2, document3, document4]
    return corpus

def preprocess(corpus: list[str]) -> list[list[str]]:
    """
    Applies:
    1. lowercase transformation
    2. leading & trailing whitespace removal
    3. removing any non-alphanumeric symbols (e.g., punctuation)
    4. splitting each documents by whitespace
    """

    result = []
    for document in tqdm(corpus, desc="Preprocessing"):
        document_lower = document.lower()
        document_alnum = re.sub(r'[^a-z0-9\s]', '', document_lower)
        tokens = document_alnum.strip().split()
        result.append(tokens)
    return result

def get_vocabulary(corpus: list[str], vocab_size: int = 2000) -> list[str]:
    """
    Tokenizes each document in the corpus and returns a list of distinct words.
    Sort the words by most frequent to least frequent. If there are
    more words than `vocab_size`, cut off the remaining infrequent words.
    The output should not contain duplicate tokens.
    Handles unseen tokens with an special <unk> token.
    """
    corpus = preprocess(corpus)

    vocabulary = []
    all_words = [word for doc in corpus for word in doc]
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(vocab_size - 1)
    for word, frequency in most_common_words:
      vocabulary.append(word)
    vocabulary.append("<unk>")

    return vocabulary

def get_vocab2idx(vocabulary: list[str]) -> dict[str, int]:
    """
    Returns a dictionary mapping vocabulary to its index (zero-indexed).
    Example input/output shown below.

    >>> get_vocab2idx(['a', 'b'])
    {'a': 0, 'b': 1}
    """
    vocab2idx = {}
    for idx, word in enumerate(vocabulary):
      vocab2idx[word] = idx

    return vocab2idx
