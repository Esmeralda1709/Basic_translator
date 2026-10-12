import os
import pickle
import numpy as np


def save_pickle(obj, filename):
    """
    Guarda un objeto usando pickle.

    Parameters
    ----------
    obj : any

    filename : str
    """

    with open(filename, "wb") as f:
        pickle.dump(obj, f)



def save_numpy(array, filename):
    """
    Guarda un ndarray.

    Parameters
    ----------
    array : ndarray

    filename : str
    """

    np.save(filename, array)


def save_model(vocab_en, vocab_es, reverse_vocab_es, emb_en, emb_es, W, directory="models"):
    """
    Guarda todos los elementos
    necesarios para el traductor.

    Parameters
    ----------
    vocab_en : dict

    vocab_es : dict

    reverse_vocab_es : dict

    emb_en : ndarray

    emb_es : ndarray

    W : ndarray
    """

    os.makedirs(directory, exist_ok=True)

    save_pickle(vocab_en, f"{directory}/vocab_en.pkl")
    save_pickle(vocab_es, f"{directory}/vocab_es.pkl")
    save_pickle( reverse_vocab_es, f"{directory}/reverse_vocab_es.pkl")

    save_numpy(emb_en, f"{directory}/emb_en.npy")
    save_numpy(emb_es, f"{directory}/emb_es.npy")
    save_numpy(W, f"{directory}/W.npy")

    print("   Modelos guardados correctamente.")