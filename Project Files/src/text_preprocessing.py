import re
import nltk
import numpy as np

nltk.download("punkt", quiet=True)


class TextPreprocessor:

    def __init__(self):
        pass

    def clean_text(self, text):
        text = str(text).lower()

        # Keep ! and ? because they carry emotion
        text = re.sub(r"[^a-zA-Z\s!?']", " ", text)

        tokens = nltk.word_tokenize(text)

        # Remove only simple articles
        skip_words = {"the", "a", "an"}

        tokens = [
            word
            for word in tokens
            if word not in skip_words and len(word) > 1
        ]

        return " ".join(tokens)

    def keyword_enhancement(self, probs, text, classes):

        text_lower = text.lower()

        emotion_keywords = {

            "frustrated": [
                "frustrated",
                "frustrating",
                "angry",
                "hate",
                "stuck",
                "wrong answer"
            ],

            "curious": [
                "why",
                "how",
                "what",
                "curious",
                "wonder",
                "interested",
                "learn"
            ],

            "confident": [
                "easy",
                "great",
                "excellent",
                "perfect",
                "solved"
            ],

            "bored": [
                "boring",
                "bored",
                "tired",
                "dull"
            ],

            "confused": [
                "confused",
                "don't understand",
                "unclear",
                "lost",
                "missing"
            ]

        }

        scores = {}

        for emotion, words in emotion_keywords.items():

            score = 0

            for word in words:

                if word in text_lower:

                    if word in [
                        "frustrated",
                        "curious",
                        "confident",
                        "bored",
                        "confused"
                    ]:
                        score += 10
                    else:
                        score += 2

            scores[emotion] = score

        if max(scores.values()) > 0:

            highest = max(scores.values())

            for emotion, score in scores.items():

                if score == highest and emotion in classes:

                    idx = classes.index(emotion)

                    probs[idx] *= (1 + score * 3)

            probs = probs / np.sum(probs)

        return probs


if __name__ == "__main__":

    processor = TextPreprocessor()

    text = "I am VERY frustrated!! I don't understand this topic."

    print("Clean Text:")
    print(processor.clean_text(text))

    classes = [
        "angry",
        "bored",
        "confused",
        "curious",
        "excited",
        "frustrated",
        "happy",
        "neutral",
        "sad",
        "surprised"
    ]

    probs = np.ones(len(classes)) / len(classes)

    enhanced_probs = processor.keyword_enhancement(
        probs,
        text,
        classes
    )

    print("\nEnhanced Probabilities:")
    print(enhanced_probs)