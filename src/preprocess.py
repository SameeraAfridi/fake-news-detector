import re, string, pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text
