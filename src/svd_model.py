from sklearn.decomposition import TruncatedSVD


def reduce_dimensions(matrix, n_components=100):
    """
    Reduce dimensionalidad usando SVD.

    Parameters
    ----------
    matrix : numpy.ndarray

    n_components : int

    Returns
    -------
    numpy.ndarray
        Embeddings reducidos
    """

    max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1)

    if n_components > max_components:
        n_components = max_components

    svd = TruncatedSVD(n_components=n_components, random_state=42)

    embeddings = svd.fit_transform(matrix)

    return embeddings


def print_svd_info(embeddings):
    """
    Imprime información básica
    de los embeddings.
    """

    print("     Información SVD")

    print(f"       - Número de vectores: {embeddings.shape[0]}")

    print(f"       - Dimensiones: {embeddings.shape[1]}")