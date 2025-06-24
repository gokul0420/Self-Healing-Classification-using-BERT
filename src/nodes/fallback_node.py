import click
from transformers import pipeline

class FallbackNode:
    """
    Either queries the user for clarification or runs a zero-shot backup model.
    """
    def __init__(self, strategy: str = "ask", threshold: float = 0.5):
        assert strategy in {"ask", "backup"}
        self.strategy = strategy
        self.zero_shot = None
        self.threshold = threshold

    def _backup(self, text):
        if self.zero_shot is None:
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        result = self.zero_shot(text, candidate_labels=["positive", "negative"])
        label, conf = result["labels"][0], result["scores"][0]
        return label.capitalize(), conf

    def __call__(self, text: str):
        if self.strategy == "ask":
            answer = click.prompt(
                "Could you clarify? Is this a 'positive' or 'negative' example",
                type=str
            )
            return answer.capitalize(), 1.0
        return self._backup(text)
