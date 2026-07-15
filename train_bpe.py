"""Train a simple byte-level BPE tokenizer up to vocab size 512.

Usage:
    python3 train_bpe.py

Reads ../data/train_corpus.txt, starts from 256 raw byte tokens,
learns 256 merges, and saves them to bpe_merges.json next to this file.
"""
import json
import os
from collections import Counter

from tqdm import tqdm


TARGET_VOCAB_SIZE = 512
BASE_VOCAB_SIZE = 256


def merge_once(tokens, pair, new_id):
    a, b = pair
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n and tokens[i] == a and tokens[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def main():
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "../data/train_corpus.txt")
    out_path = os.path.join(here, "bpe_merges.json")

    with open(data_path, "rb") as f:
        data = f.read()

    tokens = list(data)
    merges = []

    next_id = BASE_VOCAB_SIZE
    progress = tqdm(total=TARGET_VOCAB_SIZE - BASE_VOCAB_SIZE, desc="bpe")
    while next_id < TARGET_VOCAB_SIZE:
        pair_counts = Counter(zip(tokens, tokens[1:]))
        if not pair_counts:
            break

        best_pair, _ = pair_counts.most_common(1)[0]
        merges.append([best_pair[0], best_pair[1]])

        tokens = merge_once(tokens, best_pair, next_id)
        next_id += 1
        progress.update(1)

    progress.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_vocab_size": BASE_VOCAB_SIZE,
                "target_vocab_size": TARGET_VOCAB_SIZE,
                "merges": merges,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"saved {out_path}")
    print(f"learned merges: {len(merges)}")
    print(f"final vocab size: {BASE_VOCAB_SIZE + len(merges)}")


if __name__ == "__main__":
    main()
