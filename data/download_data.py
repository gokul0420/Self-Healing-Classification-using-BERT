"""
Download and split the IMDb sentiment dataset.
Run:  python -m data.download_data
"""
from datasets import load_dataset

def main():
    ds = load_dataset("imdb")
    # keep only a slice for quick experimentation; comment out for full set
    ds_small = {}
    for split in ("train", "test"):
        ds_small[split] = ds[split].shuffle(seed=42).select(range(5000))
        ds_small[split].to_json(f"data/{split}.jsonl", orient="records", lines=True)
    print("✅ 5 000 samples written to data/train.jsonl & data/test.jsonl")

if __name__ == "__main__":
    main()
