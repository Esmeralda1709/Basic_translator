Traductor Inglés-Español Basado en Espacios Vectoriales
Descripción
Este proyecto implementa un traductor léxico Inglés → Español basado en técnicas de Procesamiento de Lenguaje Natural (PLN).

El sistema aprende representaciones vectoriales de palabras a partir de un corpus paralelo bilingüe y posteriormente utiliza dichas representaciones para encontrar traducciones mediante búsqueda de vecinos más cercanos.

El flujo general del traductor es:

Corpus Paralelo ↓ Tokenización y limpieza ↓ Construcción de vocabulario ↓ Matriz de coocurrencia ↓ PPMI ↓ Reducción dimensional (SVD) ↓ Alineamiento EN → ES ↓ Búsqueda Top-K ↓ Traducción

Características
Tokenización de texto.
Eliminación de puntuación.
Eliminación de números.
Eliminación de palabras que contienen números.
Eliminación de stopwords en inglés y español.
Construcción de vocabularios.
Matrices palabra-contexto.
PPMI (Positive Pointwise Mutual Information).
Reducción dimensional mediante SVD.
Alineamiento de espacios vectoriales mediante Orthogonal Procrustes.
Traducción basada en similitud coseno.
Evaluación con Precision@K.
Estructura del Proyecto
translator_project/
│
├── data/
│   └── EN-ES.txt
│
├── models/
│   ├── vocab_en.pkl
│   ├── vocab_es.pkl
│   ├── reverse_vocab_es.pkl
│   ├── emb_en.npy
│   ├── emb_es.npy
│   └── W.npy
│
├── src/
│   ├── corpus.py
│   ├── tokenizer.py
│   ├── stopwords.py
│   ├── vocabulary.py
│   ├── cooccurrence.py
│   ├── ppmi.py
│   ├── svd_model.py
│   ├── bilingual.py
│   ├── alignment.py
│   ├── search.py
│   ├── save_model.py
│   └── load_model.py
│
├── train.py
├── translator.py
└── README.md
