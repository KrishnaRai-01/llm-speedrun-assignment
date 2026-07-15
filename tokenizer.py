"""Byte-level BPE tokenizer with a fixed 512-token vocabulary.

The first 256 ids are raw bytes. Ids 256..511 are merge tokens trained from
the provided corpus and loaded from bpe_merges.json.
"""
import json
import os


class BPETokenizer:
    vocab_size = 512

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.merge_to_id = {(a, b): 256 + i for i, (a, b) in enumerate(self.merges)}

        token_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            token_bytes.append(token_bytes[a] + token_bytes[b])
        self.token_bytes = token_bytes

    @staticmethod
    def _apply_merge(ids, pair, new_id):
        a, b = pair
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            ids = self._apply_merge(ids, pair, 256 + i)
        return ids

    def decode(self, ids):
        out = bytearray()
        for tok in ids:
            if tok < 0 or tok >= len(self.token_bytes):
                raise ValueError(f"token id out of range: {tok}")
            out.extend(self.token_bytes[tok])
        return out.decode("utf-8", errors="strict")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(p) for p in self.merges]}, f, ensure_ascii=True)


def load(path=None):
    """Return the tokenizer used by evaluate.py.

    By default loads bpe_merges.json from this file's directory.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    merges = payload.get("merges")
    if not isinstance(merges, list):
        raise ValueError("bpe_merges.json must contain a 'merges' list")

    tok = BPETokenizer(merges)
    if tok.vocab_size != 512:
        raise ValueError("tokenizer vocab_size must be 512")
    return tok
