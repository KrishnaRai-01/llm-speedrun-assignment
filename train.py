"""Baseline trainer. It WORKS and it is MEDIOCRE ON PURPOSE. Your hour goes
into changing what it does — schedule, init, optimizer, architecture,
tokenizer — inside the hard caps.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch
from tqdm import tqdm

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000
WARMUP_STEPS = 200
COSINE_STEPS = 1800
MAX_LR = 3e-3
MIN_LR = 3e-4


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step):
    if step <= WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS

    decay_step = min(step - WARMUP_STEPS, COSINE_STEPS)
    cosine_ratio = decay_step / COSINE_STEPS
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_ratio))
    return MIN_LR + cosine * (MAX_LR - MIN_LR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps == MAX_STEPS, f"run must use exactly {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1,
                            betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    losses = []
    pbar = tqdm(range(1, args.steps + 1), total=args.steps, desc="train")
    for step in pbar:
        lr = get_lr(step)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        step_time_ms = (time.time() - t0) / step * 1000
        pbar.set_postfix(loss=f"{loss.item():.4f}", step_ms=f"{step_time_ms:.0f}")
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  "
                  f"({step_time_ms:.0f} ms/step)")

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
