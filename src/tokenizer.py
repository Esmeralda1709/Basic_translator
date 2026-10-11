import re


def tokenize(text, stopwords=None):
    """
    Tokeniza un texto:
    - Convierte a minúsculas
    - Elimina puntuación
    - Elimina espacios extra
    - Elimina números
    - Elimina palabras que contengan números
    - Elimina stopwords (opcional)

    Parameters
    ----------
    text : str

    stopwords : set[str] | None

    Returns
    -------
    list[str]
    """

    if stopwords is None:
        stopwords = set()

    text = text.lower()

    text = re.sub( r"[^\w\s]", " ", text )
    text = re.sub( r"\s+", " ", text )
    text = text.strip()
    tokens = text.split()

    tokens = [
        token
        for token in tokens

        # eliminar números o tokens con números
        if not any(
            char.isdigit()
            for char in token
        )

        # eliminar stopwords
        and token not in stopwords
    ]

    return tokens