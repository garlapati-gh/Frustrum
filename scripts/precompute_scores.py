import json
import pandas as pd
from src.nlp.pipeline import process
from src.common.schemas import RawDoc

def main():
    print("Loading raw documents from data/raw/sample_news.jsonl...")
    docs = []
    try:
        with open("data/raw/sample_news.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    docs.append(RawDoc(**json.loads(line)))
    except FileNotFoundError:
        print("No raw docs found. Run python scripts/download_data.py first.")
        return

    print(f"Processing {len(docs)} documents through NLP engine...")
    signals = process(docs)

    print("Saving to data/signals.jsonl...")
    with open("data/signals.jsonl", "w") as f:
        for sig in signals:
            f.write(sig.model_dump_json() + "\n")

    print("NLP Precomputation complete! You can now run the API and Dashboard.")

if __name__ == "__main__":
    main()
