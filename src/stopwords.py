import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

ENGLISH_STOPWORDS = set(stopwords.words("english"))

SPANISH_STOPWORDS = set(stopwords.words("spanish"))