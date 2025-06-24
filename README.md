# 🛠️ Self-Healing Classification DAG using Transformers, PEFT & CLI

This project is a minimal and explainable **text classification system** built with the following:
- ✅ Hugging Face Transformers (DistilBERT)
- ✅ LoRA Fine-tuning using PEFT
- ✅ Confidence-based Self-Healing DAG (fallback logic)
- ✅ Click-powered Command Line Interface
- ✅ IMDb Dataset from Hugging Face
- ✅ Local logging and modular node design

> 🚀 This was developed as part of an interview assignment using the required tech stack.

---

## 📁 Folder Structure

```
self_healing_classifier/
├── data/                    # Dataset loading script
│   └── download_data.py
├── model/                   # Fine-tuning script and model checkpoint
│   └── finetune.py
├── src/                     # DAG logic and CLI
│   ├── cli.py
│   ├── dag.py
│   ├── utils.py
│   └── nodes/
│       ├── inference_node.py
│       ├── confidence_check.py
│       ├── fallback_node.py
│       └── __init__.py
├── logs/                    # Auto-generated logs
├── requirements.txt         # Required Python packages
└── README.md               # You're here!
```

---

## 📦 1. Installation

### ✅ Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
.venv\Scripts\activate     # On Windows
```

### ✅ Install requirements

```bash
pip install -r requirements.txt
```

If needed, update pip:

```bash
python -m pip install --upgrade pip
```

## 📥 2. Download Dataset

This script downloads the IMDb dataset from Hugging Face and selects 5,000 samples for quick testing.

```bash
python -m data.download_data
```

You will get:
- `data/train.jsonl`
- `data/test.jsonl`

## 🧠 3. Fine-tune the Model using LoRA (PEFT)

This script fine-tunes DistilBERT using LoRA (PEFT) on the IMDb dataset.

```bash
python model/finetune.py --epochs 2 --batch_size 16 --output_dir model/checkpoint
```

After training, you'll see your LoRA-adapted model saved at:

```
model/checkpoint/
```

## 🔄 4. DAG Design — How it Works

The application is structured like a LangGraph-style DAG:

1. **Inference Node**: Uses fine-tuned model for sentiment classification.
2. **Confidence Check Node**: If confidence < threshold (e.g. 0.75), triggers fallback.
3. **Fallback Node**:
   - **Mode `ask`**: Asks user for clarification.
   - **Mode `backup`**: Uses zero-shot classifier (facebook/bart-large-mnli) for fallback.

All decisions are logged in `logs/app.log`.

## 🖥️ 5. Run the CLI

Use this command to classify reviews through the CLI interface:

```bash
python -m src.cli --threshold 0.75 --fallback ask
```

You can switch fallback modes:

```bash
python -m src.cli --threshold 0.75 --fallback backup
```

Example CLI interaction:

```
User input: The movie was decent but dragged on.
[InferenceNode] Predicted label: Positive | Confidence: 61%
[ConfidenceCheckNode] Confidence too low. Triggering fallback…
Could you clarify? Is this a 'positive' or 'negative' example: negative
Final Label: Negative  (confidence ≈ 100%)
```

## 🗂️ 6. Logs

All outputs, decisions, and fallbacks are saved to:

```bash
logs/app.log
```

This helps track model behavior and fallback triggers.
