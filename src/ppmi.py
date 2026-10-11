import numpy as np


def compute_ppmi(matrix):
    """
    Calcula la matriz PPMI a partir
    de una matriz de coocurrencias.

    Parameters
    ----------
    matrix : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    total_count = np.sum(matrix)
    row_sum = np.sum(matrix, axis=1)
    col_sum = np.sum(matrix, axis=0)
    ppmi_matrix = np.zeros(matrix.shape, dtype=np.float32)
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                continue

            p_wc = ( matrix[i, j] / total_count )
            p_w = ( row_sum[i] / total_count )
            p_c = ( col_sum[j] / total_count )

            denominator = p_w * p_c
            if denominator == 0:
                continue

            pmi = np.log2( p_wc / denominator )
            ppmi_matrix[i, j] = max( 0.0, pmi )

    return ppmi_matrix


def ppmi_statistics(matrix):
    """
    Devuelve estadísticas básicas.
    """

    vals= {"shape": matrix.shape,
           "nonzero": int(np.count_nonzero(matrix)),
           "max": float(np.max(matrix)),
           "min": float(np.min(matrix)),
           "mean": float(np.mean(matrix))
        }

    return vals


def print_ppmi_statistics(matrix):
    """
    Imprime estadísticas PPMI.
    """

    stats = ppmi_statistics(matrix)

    print("     Estadísticas PPMI")

    print(f"       - Shape: {stats['shape']}")

    print(f"       - Valores no cero: {stats['nonzero']}")

    print(f"       - Máximo: {stats['max']:.4f}")

    print(f"       - Promedio: {stats['mean']:.4f}")

    print(f"       - Mínimo: {stats['min']:.4f}")