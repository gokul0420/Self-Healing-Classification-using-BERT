import logging, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

LOG_PATH = "logs/app.log"
os.makedirs("logs", exist_ok=True)

def setup_logger():
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

def load_classifier(model_dir="model/checkpoint", device=None):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    pipe = pipeline("text-classification", model=mdl, tokenizer=tok,
                    return_all_scores=True, device=0 if device=="cuda" else -1)
    return pipe

def label_and_conf(scores):
    """Return (label, confidence) from pipeline output."""
    best = max(scores, key=lambda d: d["score"])
    return best["label"], best["score"]
