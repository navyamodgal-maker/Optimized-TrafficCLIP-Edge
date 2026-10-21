import torch
import torch.nn.functional as F


class OpenSetDetector:
    """
    Detects whether a prediction belongs to a known
    traffic class or should be treated as UNKNOWN.

    UNKNOWN represents a possible zero-day/unseen class.
    """

    def __init__(
        self,
        confidence_threshold=0.60,
        entropy_threshold=1.50
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.entropy_threshold = (
            entropy_threshold
        )

    @staticmethod
    def prediction_entropy(
        probabilities
    ):
        """
        Calculate prediction entropy.
        """

        probabilities = torch.clamp(
            probabilities,
            min=1e-8
        )

        entropy = -torch.sum(
            probabilities *
            torch.log(probabilities),
            dim=1
        )

        return entropy

    def predict(
        self,
        logits
    ):
        """
        Args:
            logits:
                Model output [batch_size, num_classes]

        Returns:
            Dictionary containing prediction,
            confidence and entropy.
        """

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        entropy = self.prediction_entropy(
            probabilities
        )

        # A sample is considered known only when:
        # 1. Confidence is sufficiently high
        # 2. Entropy is sufficiently low

        known = (
            (confidence >= self.confidence_threshold)
            &
            (entropy <= self.entropy_threshold)
        )

        final_prediction = prediction.clone()

        # -1 represents UNKNOWN / ZERO-DAY
        final_prediction[
            ~known
        ] = -1

        return {
            "prediction": final_prediction,
            "confidence": confidence,
            "entropy": entropy,
            "probabilities": probabilities,
            "known": known
        }


if __name__ == "__main__":

    detector = OpenSetDetector()

    logits = torch.tensor(
        [
            [5.0, 1.0, 0.5],
            [1.0, 1.0, 1.0]
        ]
    )

    result = detector.predict(
        logits
    )

    print("Predictions :", result["prediction"])
    print("Confidence  :", result["confidence"])
    print("Entropy     :", result["entropy"])
    print("Known       :", result["known"])