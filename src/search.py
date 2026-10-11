import numpy as np


def cosine_similarity(vector_a,vector_b):
    """
    Similaridad coseno entre dos vectores.

    Parameters
    ----------
    vector_a : ndarray

    vector_b : ndarray

    Returns
    -------
    float
    """

    denominator = ( np.linalg.norm(vector_a) * np.linalg.norm(vector_b) )

    if denominator == 0:
        return 0.0

    result = float( np.dot( vector_a, vector_b  ) / denominator )

    return result


def top_k_translation(
        word,
        vocab_en,
        reverse_vocab_es,
        emb_en,
        emb_es,
        W,
        k=5):
    """
    Traduce una palabra del ingles al español
    usando vecinos más cercanos.

    Parameters
    ----------
    word : str

    vocab_en : dict

    reverse_vocab_es : dict

    emb_en : ndarray

    emb_es : ndarray

    W : ndarray

    k : int

    Returns
    -------
    list[tuple]

    Ejemplo:

    [
        ("mantequilla", 0.95),
        ("nata", 0.90),
        ...
    ]
    """

    if word not in vocab_en:
        return []

    en_idx = vocab_en[word]
    english_vector = emb_en[en_idx]
    projected_vector = (english_vector @ W)

    similarities = []

    for es_idx in range( len(emb_es) ):
        spanish_vector = emb_es[ es_idx ]

        score = cosine_similarity( projected_vector, spanish_vector )
        similarities.append( (es_idx, score) )

    similarities.sort( key=lambda x: x[1], reverse=True )

    top_results = []

    for es_idx, score in similarities[:k]:
        spanish_word = reverse_vocab_es[ es_idx ]
        top_results.append( (spanish_word, float(score)) )

    return top_results


def print_results(results, n=5):
    """
    Imprime resultados Top-K.
    """

    if len(results) == 0:
        print("   Palabra no encontrada, intente otra vez")
        return

    print(f"Top {n} traducciones:")

    for rank, (word, score) in enumerate(results, start=1):
        print(f"      {rank}. {word:<20} -> {score:.4f}")