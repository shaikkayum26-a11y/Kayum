"""
Activity 3.3
Mixed Emotion Detection

Detects multiple emotions whose confidence scores
are above a threshold (default = 15%).
"""


class MixedEmotionDetector:

    def __init__(self, threshold=0.15):
        self.threshold = threshold

    ###########################################################

    def get_mixed_emotions(self, scores):

        """
        scores example:

        {
            "Curious":0.52,
            "Confused":0.29,
            "Frustrated":0.10,
            "Bored":0.05,
            "Confident":0.04
        }
        """

        if not scores:
            return []

        # Sort highest probability first
        sorted_emotions = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary = sorted_emotions[0]

        mixed = [primary]

        for emotion, score in sorted_emotions[1:]:

            if score >= self.threshold:
                mixed.append((emotion, score))

        return mixed

    ###########################################################

    def format_output(self, mixed):

        if len(mixed) == 1:

            emotion, score = mixed[0]

            return {
                "type": "Single Emotion",
                "emotion": emotion,
                "confidence": round(score, 4)
            }

        else:

            emotions = [emotion for emotion, _ in mixed]

            primary = mixed[0]

            return {
                "type": "Mixed Emotion",
                "emotion": " + ".join(emotions),
                "confidence": round(primary[1], 4)
            }


###############################################################

if __name__ == "__main__":

    detector = MixedEmotionDetector(threshold=0.15)

    sample_scores = {

        "Curious":0.45,
        "Confused":0.31,
        "Frustrated":0.14,
        "Bored":0.06,
        "Confident":0.04

    }

    result = detector.get_mixed_emotions(sample_scores)

    output = detector.format_output(result)

    print("\nDetected Emotion")
    print("-------------------------")

    print(output)

    print("\nAll Mixed Emotions")

    for emotion, score in result:
        print(f"{emotion:12} : {score:.4f}")