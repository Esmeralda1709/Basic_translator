def load_parallel_corpus(filepath, max_pairs=None):
    """
    Carga un corpus paralelo EN-ES.

    Se espera que cada línea tenga al menos:

    ingles<TAB>espanol<TAB>...

    Returns
    -------
    list[tuple]
        [
            (texto_en, texto_es),
            ...
        ]
    """

    pairs = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if (max_pairs is not None) and (i >= max_pairs ):
                break

            columns = line.strip().split("\t")

            if len(columns) < 2:
                continue

            pairs.append( (columns[0].strip(), columns[1].strip()) )

    return pairs


def corpus_statistics(corpus):
    """
    Devuelve estadísticas básicas del corpus.
    """

    total_pairs = len(corpus)

    total_en_chars = 0
    total_es_chars = 0

    for en, es in corpus:

        total_en_chars += len(en)
        total_es_chars += len(es)

    return {
        "pairs": total_pairs,
        "english_characters": total_en_chars,
        "spanish_characters": total_es_chars
    }