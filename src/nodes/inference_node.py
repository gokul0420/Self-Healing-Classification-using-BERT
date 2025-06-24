class InferenceNode:
    """
    Runs classification using the fine-tuned model.
    Usage: label, conf = node(text)
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, text: str):
        scores = self.pipeline(text)[0]
        label, conf = max(scores, key=lambda d: d["score"]).values()
        return label, conf
