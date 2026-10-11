import pickle
import numpy as np


def load_pickle(filename):
    
    with open( filename, "rb" ) as f:
        return pickle.load(f)


def load_model( directory="models" ):
    """
    Carga todos los componentes
    necesarios para traducir.

    Returns
    -------
    tuple
    (
        vocab_en,
        vocab_es,
        reverse_vocab_es,
        emb_en,
        emb_es,
        W
    )
    """

    vocab_en = load_pickle(f"{directory}/vocab_en.pkl")
    vocab_es = load_pickle(f"{directory}/vocab_es.pkl")
    reverse_vocab_es = load_pickle(f"{directory}/reverse_vocab_es.pkl")

    emb_en = np.load(f"{directory}/emb_en.npy")
    emb_es = np.load(f"{directory}/emb_es.npy")

    W = np.load(f"{directory}/W.npy")

    return (vocab_en, vocab_es, reverse_vocab_es, emb_en, emb_es, W)


def print_model_info(vocab_en, vocab_es, emb_en, emb_es, W):
    """
    Muestra información básica
    del modelo cargado.
    """

    print("\n-- Modelo Cargado --")

    print(f"   Vocabulario EN: {len(vocab_en)}")

    print(f"   Vocabulario ES: {len(vocab_es)}")

    print(f"   Embeddings EN: {emb_en.shape}")

    print(f"   Embeddings ES: {emb_es.shape}")

    print(f"   Matriz W: {W.shape}")