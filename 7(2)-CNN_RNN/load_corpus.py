from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    for item in dataset:
        text = item["verse_text"]
        if text.strip():
            corpus.append(text)
    return corpus