import numpy as np


def build_training_matrices(seed_dictionary, vocab_en, vocab_es, emb_en, emb_es):
    """
    Construye las matrices X e Y para
    entrenar la transformación EN -> ES.

    Parameters
    ----------
    seed_dictionary : list[tuple]
        [
            ("butter", "mantequilla"),
            ("cream", "nata"),
            ...
        ]

    vocab_en : dict
    vocab_es : dict

    emb_en : ndarray
    emb_es : ndarray

    Returns
    -------
    tuple
        (X, Y)
    """

    X = []
    Y = []
    for en_word, es_word in seed_dictionary:
        if (en_word not in vocab_en) or (es_word not in vocab_es):
            continue

        en_idx = vocab_en[en_word]
        es_idx = vocab_es[es_word]

        X.append( emb_en[en_idx] )
        Y.append( emb_es[es_idx] )

    X = np.array( X, dtype=np.float32 )
    Y = np.array( Y, dtype=np.float32 )

    return X, Y


def learn_mapping(X, Y):
    """
    Aprende transformación usando
    Orthogonal Procrustes.

    Parameters
    ----------
    X : ndarray

    Y : ndarray

    Returns
    -------
    ndarray
        Matriz de transformación W
    """

    M = X.T @ Y

    U, _, Vt = np.linalg.svd(M)

    W = U @ Vt

    return W


def alignment_error(X, Y, W):
    """
    Error cuadrático medio.

    Parameters
    ----------
    X : ndarray

    Y : ndarray

    W : ndarray

    Returns
    -------
    float
    """

    X_mapped = X @ W

    mse = np.mean( (X_mapped - Y) ** 2 )

    return float(mse)


def mean_cosine_similarity(X, Y, W):
    """
    Similaridad coseno promedio.

    Parameters
    ----------
    X : ndarray

    Y : ndarray

    W : ndarray

    Returns
    -------
    float
    """

    X_mapped = X @ W

    similarities = []
    for i in range( len(X_mapped) ):
        x = X_mapped[i]
        y = Y[i]

        denominator = ( np.linalg.norm(x) * np.linalg.norm(y) )

        if denominator == 0:
            continue

        cosine = ( np.dot(x, y) / denominator )
        similarities.append( cosine )

    if len(similarities) == 0:
        return 0.0

    return float(np.mean(similarities))


def print_alignment_info(X, Y, W):
    """
    Imprime estadísticas del modelo.
    """

    print("  Estadísitcas del modelo")

    print(f"     - Tamaño X: {X.shape}")
    print(f"     - Tamaño Y: {Y.shape}")
    print(f"     - Tamaño W: {W.shape}")

    mse = alignment_error(X, Y, W)
    print(f"     - MSE: {mse:.6f}")

    mcs = mean_cosine_similarity(X, Y, W)
    print(f"     - Cosine Score: {mcs:.6f}")