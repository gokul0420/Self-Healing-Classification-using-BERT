class ConfidenceCheckNode:
    """
    Decides whether confidence is sufficient.
    Returns True if fallback is required.
    """
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def __call__(self, confidence: float) -> bool:
        return confidence < self.threshold
