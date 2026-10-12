import numpy as np


def build_cooccurrence(docs, vocab, window_size=4):
    """
    Construye una matriz palabra-contexto.

    Parameters
    ----------
    docs : list[list[str]]

    vocab : dict
        palabra -> índice

    window_size : int

    Returns
    -------
    numpy.ndarray
    """

    vocab_size = len(vocab)

    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for doc in docs:
        doc_length = len(doc)

        for i, word in enumerate(doc):
            if word not in vocab:
                continue

            word_id = vocab[word]
            left = max(0, i - window_size)
            right = min(doc_length, i + window_size + 1 )

            for j in range(left, right):
                if i == j:
                    continue

                context_word = doc[j]

                if context_word not in vocab:
                    continue

                context_id = vocab[context_word]

                matrix[word_id, context_id] += 1.0

    return matrix


def matrix_density(matrix):
    """
    Calcula el porcentaje de celdas
    distintas de cero.

    Parameters
    ----------
    matrix : numpy.ndarray

    Returns
    -------
    float
    """

    non_zero = np.count_nonzero(matrix)
    total = matrix.size
    resul = (non_zero / total) * 100.0

    return resul


def print_matrix_info(matrix):
    """
    Imprime estadísticas básicas.
    """

    print("     Información de matriz:")

    print(f"       - Shape: {matrix.shape}")

    print(f"       - Elementos no cero: {np.count_nonzero(matrix)}")

    print(f"       - Densidad: {matrix_density(matrix):.4f}%")