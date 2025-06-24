import logging
from src.nodes.inference_node import InferenceNode
from src.nodes.confidence_check_node import ConfidenceCheckNode
from src.nodes.fallback_node import FallbackNode
from src.utils import setup_logger, load_classifier


class ClassificationDAG:
    def __init__(self, model_dir="model/checkpoint",
                 conf_threshold=0.75,
                 fallback_strategy="ask"):
        setup_logger()
        self.logger = logging.getLogger(__name__)
        pipe = load_classifier(model_dir)
        self.infer   = InferenceNode(pipe)
        self.conf_ck = ConfidenceCheckNode(conf_threshold)
        self.fallback= FallbackNode(fallback_strategy)

    def run(self, text: str):
        label, conf = self.infer(text)
        self.logger.info(f"[InferenceNode] '{text}' -> {label} ({conf:.2f})")
        print(f"[InferenceNode] Predicted label: {label} | Confidence: {conf:.0%}")

        if self.conf_ck(conf):
            print("[ConfidenceCheckNode] Confidence too low. Triggering fallback…")
            self.logger.info("Confidence below threshold -- invoking fallback")
            label, conf = self.fallback(text)
            self.logger.info(f"[FallbackNode] Resolved label='{label}' conf={conf:.2f}")

        print(f"Final Label: {label}  (confidence ≈ {conf:.0%})")
        self.logger.info(f"Final label decided: {label}\n"+"-"*50)
