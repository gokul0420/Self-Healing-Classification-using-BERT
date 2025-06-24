# Self-Healing Classification DAG 🩹🤖

A minimal LangGraph-style workflow that fine-tunes DistilBERT on IMDb,
then routes low-confidence predictions through a fallback (ask-the-user or
zero-shot backup).

## 1. Setup

```bash
git clone <your-repo> && cd self_healing_classifier
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
