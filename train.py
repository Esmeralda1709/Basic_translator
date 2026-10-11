from src.tokenizer import tokenize
from src.save_model import save_model
from src.bilingual import build_seed_dictionary
from src.vocabulary import build_vocab, invert_vocab
from src.ppmi import compute_ppmi, print_ppmi_statistics
from src.svd_model import reduce_dimensions, print_svd_info
from src.corpus import load_parallel_corpus, corpus_statistics
from src.stopwords import ENGLISH_STOPWORDS, SPANISH_STOPWORDS
from src.cooccurrence import build_cooccurrence, print_matrix_info
from src.alignment import build_training_matrices, learn_mapping, print_alignment_info


print("\n=========== Entrenamiento del traductor Ingles al Español ===========")

# 1. Cargar base de datos
print("\n[1/9] Cargando datos")
corpus = load_parallel_corpus( "data/EN-ES.txt", max_pairs=200000)

print(f"  {corpus_statistics(corpus)}")


# 2. Limpieza
print("\n[2/9] Tokenizando datos")

en_docs = []
es_docs = []

for en_text, es_text in corpus:
    en_docs.append( tokenize(en_text, ENGLISH_STOPWORDS))
    es_docs.append( tokenize(es_text, SPANISH_STOPWORDS) )


# 3. Generar voca ulario
print("\n[3/9] Construyendo vocabulario")

vocab_en = build_vocab( en_docs )
vocab_es = build_vocab( es_docs )

reverse_vocab_es = invert_vocab(vocab_es)


# 4. Coocurrencia
print("\n[4/9] Construyendo matrices de coocurrencia")

cooc_en = build_cooccurrence( en_docs, vocab_en )
cooc_es = build_cooccurrence( es_docs, vocab_es )

print("  - Ingles")
print_matrix_info(cooc_en)

print("\n  - Español")
print_matrix_info(cooc_es)


# 5. PPMI
print("\n[5/9] Calculando PPMI...")

ppmi_en = compute_ppmi( cooc_en )
ppmi_es = compute_ppmi( cooc_es )

print("  - Ingles")
print_ppmi_statistics(ppmi_en)

print("\n  - Español")
print_ppmi_statistics(ppmi_es)


# 6. SVD
print("\n[6/9] Aplicando SVD...")

emb_en = reduce_dimensions( ppmi_en )
emb_es = reduce_dimensions( ppmi_es )

print("  - Ingles")
print_svd_info(emb_en)

print("\n  - Español")
print_svd_info(emb_es)


# 7. Diccionario
print("\n[7/9] Generando diccionario")

seed_dictionary = build_seed_dictionary(en_docs, es_docs, vocab_en, vocab_es)


# 8. Entrenamiento
print("\n[8/9] Entrenamiento del modelo")

X_train, Y_train = build_training_matrices( seed_dictionary, vocab_en, vocab_es,
                                            emb_en, emb_es )

W = learn_mapping( X_train, Y_train)

print_alignment_info( X_train, Y_train, W )


# 9. Guardar modelo
print("\n[9/9] Guardando modelo")

save_model(vocab_en, vocab_es, reverse_vocab_es, emb_en, emb_es, W)


print("\nEntrenamiento finalizado.")