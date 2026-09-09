from dataclasses import dataclass


@dataclass
class ModelSummary:
    loss: float
    accuracy: float
    f1_score: float

    def print(self):
        print(
            "Model Summary:\n"
            f"Loss: {self.loss:.4f}\n"
            f"Accuracy: {self.accuracy:.2f} %\n"
            f"F1 Score: {self.f1_score:.2f} %\n"
        )
