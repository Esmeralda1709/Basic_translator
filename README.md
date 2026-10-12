# 🌐 Traductor Inglés-Español Basado en Espacios Vectoriales

<p align="center">
  <strong>🤖 Traductor léxico bilingüe mediante Procesamiento de Lenguaje Natural</strong>
  <br>
  Inglés 🇬🇧 ➜ Español 🇪🇸
</p>

---

## 📌 Descripción

Este proyecto implementa un **traductor léxico Inglés → Español** basado en técnicas de **Procesamiento de Lenguaje Natural (PLN)**.

El sistema aprende representaciones vectoriales de palabras a partir de un corpus paralelo bilingüe y utiliza dichas representaciones para encontrar posibles traducciones mediante una búsqueda de vecinos más cercanos.

El objetivo es identificar palabras en español que tengan una representación vectorial similar a la palabra de entrada en inglés, utilizando técnicas de análisis distribucional y alineamiento de espacios vectoriales.

### 🔄 Flujo general del sistema

```text
📚 Corpus paralelo bilingüe
          ↓
🧹 Tokenización y limpieza
          ↓
📖 Construcción de vocabularios
          ↓
🔢 Matriz de coocurrencia
          ↓
📊 Cálculo de PPMI
          ↓
📉 Reducción dimensional mediante SVD
          ↓
🔗 Alineamiento de espacios vectoriales EN → ES
          ↓
🔎 Búsqueda de vecinos más cercanos (Top-K)
          ↓
💬 Traducción de palabras
```

---

## ✨ Características principales

- 🧹 Tokenización y limpieza de texto.
- 🔡 Conversión de palabras a minúsculas.
- ✂️ Eliminación de signos de puntuación.
- 🔢 Eliminación de números y palabras que contienen números.
- 🚫 Eliminación de *stopwords* en inglés y español.
- 📚 Construcción de vocabularios bilingües.
- 🔢 Generación de matrices de coocurrencia palabra-contexto.
- 📊 Cálculo de PPMI (*Positive Pointwise Mutual Information*).
- 📉 Reducción dimensional mediante SVD (*Singular Value Decomposition*).
- 🔗 Alineamiento de espacios vectoriales mediante Orthogonal Procrustes.
- 🎯 Búsqueda de traducciones mediante similitud coseno.
- 🔎 Generación de predicciones mediante Top-K.
- 📈 Evaluación de resultados mediante Precision@K.
- 💾 Guardado y carga de los modelos entrenados.

---

## 🗂️ Estructura del proyecto

```text
📁 translator_project/
 │
 ├── 📁 data/
 │   └── EN-ES.txt
 │
 ├── 📁 models/
 │   ├── vocab_en.pkl
 │   ├── vocab_es.pkl
 │   ├── reverse_vocab_es.pkl
 │   ├── emb_en.npy
 │   ├── emb_es.npy
 │   └── W.npy
 │
 ├── 📁 src/
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
```

### 📂 Descripción de los directorios

- `data/`: contiene el corpus paralelo bilingüe.
- `models/`: almacena los vocabularios, embeddings y la matriz de transformación generados durante el entrenamiento.
- `src/`: contiene los módulos encargados del procesamiento del corpus y la construcción de representaciones vectoriales.
- `train.py`: ejecuta el proceso de entrenamiento.
- `translator.py`: permite consultar traducciones.
- `README.md`: contiene la documentación del proyecto.

---

## 🛠️ Requisitos

Antes de ejecutar el proyecto, asegúrate de contar con los siguientes componentes:

- 🐍 **Python 3.10 o superior**
- 🔢 NumPy
- 🧮 SciPy
- 🤖 Scikit-learn
- 📝 NLTK

### 📦 Instalación de dependencias

Abre una terminal en el directorio raíz del proyecto y ejecuta:

```bash
pip install numpy scipy scikit-learn nltk
```

💡 Se recomienda utilizar un entorno virtual para mantener las dependencias aisladas de otras instalaciones de Python.

---

## ⚙️ Configuración inicial

El proyecto utiliza las listas de *stopwords* proporcionadas por NLTK para filtrar palabras comunes en inglés y español.

Para descargar este recurso, ejecuta una sola vez el siguiente código:

```python
import nltk

nltk.download("stopwords")
```

✅ Una vez descargadas, las listas estarán disponibles para su uso durante el procesamiento del corpus.

---

## 📚 Corpus bilingüe

El sistema utiliza un corpus paralelo Inglés-Español almacenado en el archivo `data/EN-ES.txt`.

### 📝 Formato esperado

Cada línea debe contener una palabra o frase en inglés y su correspondiente traducción al español, separadas por un tabulador (`TAB`).

```text
English text<TAB>Spanish text
```

### 💬 Ejemplo

```text
Butter	Mantequilla
Cream	Nata
Formula	Fórmula
```

📍 **Ubicación del archivo:**

```text
data/EN-ES.txt
```

⚠️ **Importante:** el corpus debe respetar el formato esperado por el módulo de carga. Si el programa utiliza tabuladores como separadores, no deben sustituirse por espacios u otros caracteres.

La calidad y el tamaño del corpus influyen directamente en los vocabularios, las representaciones vectoriales y las traducciones obtenidas.

---

## 🧹 Limpieza y procesamiento de datos

Antes de construir las representaciones vectoriales, el sistema procesa el texto para reducir variaciones innecesarias y eliminar elementos que no se utilizarán en el análisis.

### 1️⃣ Conversión a minúsculas

Se convierten las palabras a minúsculas para evitar que una misma palabra aparezca como entrada diferente debido al uso de mayúsculas.

**Ejemplo:**

```text
Butter
```

⬇️ Resultado:

```text
butter
```

### 2️⃣ Eliminación de puntuación

Se eliminan los signos de puntuación del texto.

**Ejemplo:**

```text
Commission Regulation (EC)
```

⬇️ Resultado:

```text
commission regulation ec
```

### 3️⃣ Eliminación de números

Se eliminan las expresiones numéricas que no se consideran útiles para el vocabulario.

**Ejemplo:**

```text
1788/2004
```

⬇️ Resultado: expresión eliminada.

### 4️⃣ Eliminación de palabras que contienen números

También se descartan palabras que contienen caracteres numéricos.

**Ejemplo:**

```text
150th
```

⬇️ Resultado: palabra eliminada.

### 5️⃣ Eliminación de *stopwords*

Se filtran palabras frecuentes que suelen aportar poca información para la tarea de representación distribucional.

Algunos ejemplos en inglés son:

```text
the
of
and
for
```

⬇️ Estas palabras se eliminan de acuerdo con las listas de *stopwords* utilizadas durante el procesamiento.

---

## 🚀 Entrenamiento del modelo

El entrenamiento genera los recursos necesarios para realizar las consultas de traducción.

Desde el directorio raíz del proyecto, ejecuta:

```bash
python train.py
```

### 🔬 Etapas del entrenamiento

1. 📥 Carga del corpus paralelo bilingüe.
2. 🧹 Tokenización y limpieza del texto.
3. 📚 Construcción de los vocabularios en inglés y español.
4. 🔢 Construcción de las matrices de coocurrencia.
5. 📊 Cálculo de las matrices PPMI.
6. 📉 Reducción dimensional mediante SVD.
7. 🔗 Construcción del diccionario semilla bilingüe.
8. 🧠 Aprendizaje de la matriz de transformación entre los espacios vectoriales.
9. 💾 Guardado de los modelos y recursos generados.

### 📦 Archivos generados

Al finalizar correctamente, se espera que el directorio `models/` contenga:

```text
📁 models/
 ├── vocab_en.pkl
 ├── vocab_es.pkl
 ├── reverse_vocab_es.pkl
 ├── emb_en.npy
 ├── emb_es.npy
 └── W.npy
```

✅ Estos archivos se utilizan posteriormente para realizar consultas sin repetir el entrenamiento, siempre que se conserven y sean compatibles con el modelo utilizado.

---

## 💬 Traducción de palabras

Una vez entrenado el modelo, ejecuta:

```bash
python translator.py
```

El programa permite introducir una palabra en inglés y obtener las mejores traducciones candidatas en español, ordenadas según su similitud vectorial.

### 🧪 Ejemplo de ejecución

**Entrada:**

```text
Palabra en ingles: butter
```

**Posible salida:**

```text
Top 5 traducciones:
    1. mantequilla          -> 0.8945
    2. nata                 -> 0.8598
    3. desnatada            -> 0.8270
    4. concentrada          -> 0.8152
    5. leche                -> 0.7475
```

🎯 La primera predicción corresponde a la palabra candidata con mayor similitud según el método de búsqueda implementado.

*Nota: los resultados mostrados son ilustrativos. Las traducciones y puntuaciones reales dependen del corpus, del entrenamiento y de los parámetros utilizados.*

---

## 💾 Modelos generados

Durante el entrenamiento se generan archivos que almacenan los vocabularios y las representaciones vectoriales utilizadas por el sistema.

| 📄 Archivo | 📝 Descripción |
|---|---|
| `vocab_en.pkl` | Relaciona cada palabra en inglés con su índice. |
| `vocab_es.pkl` | Relaciona cada palabra en español con su índice. |
| `reverse_vocab_es.pkl` | Relaciona cada índice con su palabra en español. |
| `emb_en.npy` | Matriz de embeddings del vocabulario inglés. |
| `emb_es.npy` | Matriz de embeddings del vocabulario español. |
| `W.npy` | Matriz de transformación para alinear los espacios vectoriales. |

Los archivos `.pkl` almacenan estructuras de datos serializadas, mientras que los archivos `.npy` almacenan matrices numéricas en formato NumPy.

---

## 🧩 Función de cada archivo

| 📄 Archivo | ⚙️ Función |
|---|---|
| `corpus.py` | 📥 Carga el corpus paralelo bilingüe. |
| `tokenizer.py` | 🧹 Realiza la tokenización y limpieza del texto. |
| `stopwords.py` | 🚫 Gestiona las listas de *stopwords*. |
| `vocabulary.py` | 📚 Construye los vocabularios y las correspondencias entre palabras e índices. |
| `cooccurrence.py` | 🔢 Construye las matrices de coocurrencia palabra-contexto. |
| `ppmi.py` | 📊 Calcula las matrices PPMI. |
| `svd_model.py` | 📉 Reduce la dimensionalidad mediante SVD. |
| `bilingual.py` | 🔗 Construye el diccionario semilla bilingüe. |
| `alignment.py` | 🧠 Aprende la transformación para alinear los espacios vectoriales. |
| `search.py` | 🔎 Busca las traducciones candidatas mediante similitud vectorial. |
| `save_model.py` | 💾 Guarda los recursos y modelos entrenados. |
| `load_model.py` | 📂 Carga los recursos y modelos guardados. |
| `train.py` | 🚀 Coordina las etapas de entrenamiento. |
| `translator.py` | 💬 Ejecuta la interfaz de consulta del traductor. |

---

## ⚠️ Limitaciones

- 🔤 El sistema está diseñado para traducir palabras individuales, no oraciones completas.
- 📖 No considera necesariamente el contexto gramatical de una oración.
- 📚 La calidad de las traducciones depende del tamaño, la calidad y la cobertura del corpus bilingüe.
- 🔍 Las palabras poco frecuentes pueden tener representaciones vectoriales insuficientes.
- 🔗 El alineamiento depende de la calidad del diccionario semilla y de las representaciones aprendidas.
- 🎯 La búsqueda por similitud vectorial puede devolver palabras relacionadas semánticamente que no sean traducciones exactas.
- 💻 El entrenamiento puede requerir una cantidad considerable de memoria y tiempo de procesamiento cuando se utilizan corpus grandes.
- 📊 La evaluación depende de la calidad del conjunto de palabras de prueba y de la métrica implementada.

---

## 🛠️ Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| 🐍 **Python** | Lenguaje de programación principal. |
| 🔢 **NumPy** | Manipulación de matrices y operaciones numéricas. |
| 🧮 **SciPy** | Operaciones científicas y álgebra lineal. |
| 🤖 **Scikit-learn** | Herramientas de aprendizaje automático y procesamiento de vectores, según las funciones utilizadas. |
| 📝 **NLTK** | Procesamiento de texto y listas de *stopwords*. |

---

## 👩‍💻 Autor

Proyecto desarrollado para la materia de **Procesamiento de Lenguaje Natural**.

🎓 **Universidad de Guanajuato** 🇲🇽

---

⭐ *Gracias por visitar este proyecto. ¡Espero que te resulte interesante!*
