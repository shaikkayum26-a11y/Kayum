import os
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class BERTEmotionClassifier:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.tokenizer = None
        self.id2label = {}
        self.emotion_labels = []

    ###########################################################
    def load_model(self):

        import joblib

        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        MODEL_PATH = os.path.join(
            BASE_DIR,
            "models",
            "bert"
        )

        print("Loading BERT model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        self.model.to(self.device)
        self.model.eval()

        #######################################################
        # Load Label Encoder
        #######################################################

        label_path = os.path.join(
            MODEL_PATH,
            "label_encoder.pkl"
        )

        if os.path.exists(label_path):

            label_encoder = joblib.load(label_path)

            if hasattr(label_encoder, "classes_"):

                self.id2label = {
                    i: label
                    for i, label in enumerate(label_encoder.classes_)
                }

            else:

                print("Warning: label_encoder.pkl is not fitted.")

                self.id2label = {
                    i: str(i)
                    for i in range(self.model.config.num_labels)
                }

        else:

            print("Warning: label_encoder.pkl not found.")

            if self.model.config.id2label:

                self.id2label = {
                    int(k): v
                    for k, v in self.model.config.id2label.items()
                }

            else:

                self.id2label = {
                    i: str(i)
                    for i in range(self.model.config.num_labels)
                }

        self.emotion_labels = list(self.id2label.values())

        print("BERT Model Loaded Successfully.")

      ###########################################################
    def predict(self, text):

        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_model() first."
            )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=1
            ).cpu().numpy()[0]

        ####################################################
        # Class Weights
        ####################################################

        class_weights = np.ones(len(probs))

        weighted_probs = probs * class_weights

        weighted_probs = weighted_probs / weighted_probs.sum()

        emotion_idx = np.argmax(weighted_probs)

        emotion = self.id2label[emotion_idx]

        confidence = float(weighted_probs[emotion_idx])

        scores = {
            self.id2label[i]: round(float(weighted_probs[i]), 4)
            for i in range(len(weighted_probs))
        }

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "scores": scores,
            "cleaned_text": text.strip()
        }


#############################################################

if __name__ == "__main__":

    classifier = BERTEmotionClassifier()

    classifier.load_model()

    text = input("\nEnter Student Text : ")

    result = classifier.predict(text)

    print("\nPrediction")
    print("---------------------------")
    print("Emotion     :", result["emotion"])
    print("Confidence  :", result["confidence"])

    print("\nAll Scores")

    for emotion, score in result["scores"].items():
        print(f"{emotion:12} : {score:.4f}")