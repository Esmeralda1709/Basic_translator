from collections import Counter


def build_vocab(documents, min_freq=5):
    """
    Construye un vocabulario a partir de documentos tokenizados.

    Parameters
    ----------
    documents : list[list[str]]

    min_freq : int

    Returns
    -------
    dict
        palabra -> índice
    """

    counter = Counter()

    for doc in documents:
        counter.update(doc)

    vocab = {}
    idx = 0

    for word, freq in sorted(counter.items()):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def invert_vocab(vocab):
    """
    Convierte:

    palabra -> índice

    a:

    índice -> palabra
    """

    reverse_vocab = {}

    for word, idx in vocab.items():
        reverse_vocab[idx] = word

    return reverse_vocab