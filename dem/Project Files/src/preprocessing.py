import os
import re
import string
import joblib
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("punkt")
nltk.download("stopwords")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "emotion_text_dataset.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.lower().str.strip()

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["emotion"])

joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

tokenizer = Tokenizer(
    num_words=30000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["clean_text"])

joblib.dump(
    tokenizer,
    os.path.join(MODEL_DIR, "tokenizer.pkl")
)

sequences = tokenizer.texts_to_sequences(df["clean_text"])

padded_sequences = pad_sequences(
    sequences,
    maxlen=80,
    padding="post",
    truncating="post"
)

np.save(
    os.path.join(MODEL_DIR, "padded_sequences.npy"),
    padded_sequences
)

df.to_csv(
    os.path.join(MODEL_DIR, "processed_dataset.csv"),
    index=False
)

print("Dataset Shape:", df.shape)
print("Sequence Shape:", padded_sequences.shape)
print("Classes:", list(encoder.classes_))
print("Vocabulary Size:", len(tokenizer.word_index))
print("Preprocessing Completed Successfully")