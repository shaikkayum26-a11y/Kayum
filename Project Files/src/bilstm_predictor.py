import os
import re
import joblib
import numpy as np
import tensorflow as tf

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 80


class EmotionPredictor:

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        MODEL_PATH = os.path.join(
            BASE_DIR,
            "models",
            "bltsm",
            "bilstm_student_adaptive.keras"
        )

        TOKENIZER_PATH = os.path.join(
            BASE_DIR,
            "models",
            "bltsm",
            "tokenizer.pkl"
        )

        LABEL_ENCODER_PATH = os.path.join(
            BASE_DIR,
            "models",
            "bltsm",
            "label_encoder.pkl"
        )

        print("Loading BiLSTM model...")

        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)

        except Exception:
            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

        print("Loading tokenizer...")
        self.tokenizer = joblib.load(TOKENIZER_PATH)

        print("Loading label encoder...")
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        if not hasattr(label_encoder, "classes_"):
            raise ValueError(
                "label_encoder.pkl is not fitted.\n"
                "Please recreate it using LabelEncoder().fit(emotion_labels)."
            )

        self.classes = list(label_encoder.classes_)

        print("Model Loaded Successfully.")

    ############################################################

    def clean_text(self, text):

        text = str(text).lower()

        text = re.sub(r"[^a-zA-Z\s!?']", " ", text)

        tokens = word_tokenize(text)

        skip_words = {"the", "a", "an"}

        tokens = [
            token
            for token in tokens
            if token not in skip_words and len(token) > 1
        ]

        return " ".join(tokens)

    ############################################################

    def predict(self, text):

        cleaned = self.clean_text(text)

        if cleaned.strip() == "":
            cleaned = text.lower()

        sequence = self.tokenizer.texts_to_sequences([cleaned])

        if len(sequence) == 0 or len(sequence[0]) == 0:

            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "scores": {},
                "cleaned_text": cleaned
            }

        padded = pad_sequences(
            sequence,
            maxlen=MAX_SEQ_LEN,
            padding="post"
        )

        probs = self.model.predict(
            padded,
            verbose=0
        )

        probs = np.array(probs).flatten()

        # Softmax Normalization
        probs = np.exp(probs) / np.sum(np.exp(probs))

        emotion_idx = np.argmax(probs)

        emotion = self.classes[emotion_idx]

        confidence = float(probs[emotion_idx])

        scores = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "scores": scores,
            "cleaned_text": cleaned
        }


############################################################

if __name__ == "__main__":

    predictor = EmotionPredictor()

    text = input("\nEnter Student Text: ")

    result = predictor.predict(text)

    print("\n========== Prediction ==========")

    print("Cleaned Text :", result["cleaned_text"])

    print("Emotion      :", result["emotion"])

    print("Confidence   :", result["confidence"])

    print("\nProbability Scores")

    for emotion, score in result["scores"].items():
        print(f"{emotion:15} : {score:.4f}")