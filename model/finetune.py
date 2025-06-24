"""
LoRA fine-tuning of DistilBERT on IMDb.
Run (GPU recommended):
  python model/finetune.py --epochs 2 --batch_size 16 --output_dir model/checkpoint
"""
import argparse, os, torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", default="model/checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- Load trimmed IMDb splits ----------
    ds_train = load_dataset("json", data_files="data/train.jsonl")["train"]
    ds_test = load_dataset("json", data_files="data/test.jsonl")["train"]

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    ds_train = ds_train.map(tokenize, batched=True)
    ds_test = ds_test.map(tokenize, batched=True)

    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ---------- Base model ----------
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # ---------- LoRA config (✅ now with target_modules) ----------
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()

    # ---------- Training ----------
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir=args.output_dir,
        logging_dir="logs/hf",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
    )
    trainer.train()

    # ---------- Save ----------
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
