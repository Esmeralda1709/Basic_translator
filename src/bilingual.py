def extract_seed_dictionary(en_docs, es_docs):
    """
    Extrae pares bilingües simples a partir
    de documentos ya tokenizados.

    Solo se conservan pares donde tanto
    el lado inglés como el español tienen
    exactamente una palabra.

    Parameters
    ----------
    en_docs : list[list[str]]

    es_docs : list[list[str]]

    Returns
    -------
    list[tuple]
    """

    seed_dictionary = []

    for en_tokens, es_tokens in zip( en_docs, es_docs ):
        if (len(en_tokens) == 1) and (len(es_tokens) == 1):

            seed_dictionary.append( (en_tokens[0], es_tokens[0]) )

    return seed_dictionary


def remove_duplicates(seed_pairs):
    """
    Elimina pares repetidos.

    Parameters
    ----------
    seed_pairs : list[tuple]

    Returns
    -------
    list[tuple]
    """

    unique_pairs = list( set(seed_pairs) )

    unique_pairs.sort()

    return unique_pairs


def filter_seed_dictionary(seed_pairs, vocab_en, vocab_es):
    """
    Conserva únicamente pares que
    existen en ambos vocabularios.

    Parameters
    ----------
    seed_pairs : list[tuple]

    vocab_en : dict

    vocab_es : dict

    Returns
    -------
    list[tuple]
    """

    filtered = []

    for en_word, es_word in seed_pairs:
        if (en_word in vocab_en) and (es_word in vocab_es):

            filtered.append( (en_word, es_word) )

    return filtered


def build_seed_dictionary(en_docs, es_docs, vocab_en, vocab_es):
    """
    Pipeline completo para crear
    el diccionario semilla.

    Parameters
    ----------
    en_docs : list[list[str]]

    es_docs : list[list[str]]

    vocab_en : dict

    vocab_es : dict

    Returns
    -------
    list[tuple]
    """

    seed_pairs = extract_seed_dictionary(en_docs, es_docs)

    seed_pairs = remove_duplicates(seed_pairs)

    seed_pairs = filter_seed_dictionary(seed_pairs, vocab_en, vocab_es)

    return seed_pairs