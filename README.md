# LLM Speedrun Assignment

**Author:** Rai Krishna RajendraKumar (IIT Kharagpur)
**Final Score:** 1.7256 bpb (Bits Per Byte)

## Overview
This repository contains my optimized submission for the 2,000-Step LLM Speedrun assignment. The objective was to take a deliberately mediocre, baseline GPT training script written in pure PyTorch and optimize it to achieve the lowest possible validation loss on a hidden text file. 

The primary challenge of this assignment was extracting the maximum possible learning signal under extremely strict hardware and budget constraints.

## Hard Constraints
This submission strictly adheres to the following assignment caps:
* **Parameter Limit:** Maximum of 2,000,000 total parameters.
* **Compute Budget:** Exactly 2,000 optimizer steps.
* **Hardware:** 100% CPU execution (no GPU/CUDA permitted).
* **Dependencies:** Pure PyTorch, NumPy, and standard library only. (No pre-trained weights, HuggingFace Transformers, Flash-Attention, or Mamba-SSM).
* **Data Limit:** Model and tokenizer trained exclusively on the provided `train_corpus.txt`.

## Final Results & Specifications
* **Validation Score:** 1.7256 bpb
* **Total Parameters:** 1,926,592
* **Total Steps:** 2,000

## Key Optimizations
To move the baseline score (typically 3.5 - 4.5 bpb) down to 1.7256 bpb, I implemented a series of architectural and training loop upgrades:

### 1. Tokenizer Upgrade (Byte Pair Encoding)
The baseline byte-level tokenizer was highly inefficient for the Hindi/Devanagari text in the corpus, often using 3 tokens per character. I wrote a custom BPE tokenizer (`train_bpe.py`) to compress the text into a 512-token vocabulary. This effectively increased the model's context horizon while maintaining a strict byte-level fallback to ensure 100% lossless encoding for arbitrary UTF-8 text.

### 2. Architectural Changes
To maximize the 2M parameter budget, I made the following modifications in `model.py`:
* **Tied Weights:** Tied the token embedding and final output layer weights. This freed up massive parameter overhead, allowing me to deepen the network to 4 layers and 6 attention heads.
* **RMSNorm:** Replaced standard LayerNorm with RMSNorm to reduce computational friction and speed up CPU execution.
* **SwiGLU:** Swapped standard GELU activations for SwiGLU in the MLP blocks to provide better non-linearity and capacity per parameter.

### 3. Optimization Strategy
A constant learning rate wastes the 2,000-step budget. In `train.py`, I implemented:
* **AdamW:** Switched to AdamW with a weight decay of 0.1 for better regularization.
* **Cosine Annealing with Warmup:** Added a learning rate schedule with 200 steps of linear warmup to prevent early divergence, followed by cosine decay down to the 2,000th step to smoothly settle into a sharp local minimum.
* **Gradient Clipping:** Capped gradients at 1.0 to protect against instability during the warmup phase.

## Repository Structure
* `model.py`: The core GPT architecture featuring RMSNorm and SwiGLU.
* `train.py`: The training loop with the AdamW optimizer and Cosine schedule.
* `tokenizer.py`: The custom BPE tokenizer interface.
* `train_bpe.py`: Script to generate the BPE merges from the training corpus.
* `evaluate.py`: The unmodified official scoring script.
* `bpe_merges.json`: The learned token merges (artifact).
* `ckpt.pt`: The final 2,000-step model checkpoint.
* `RUNLOG.md`: Documentation of the scientific method and experimental progression.
* `NOTES.md`: A brief justification of the final configuration.
* `SUMMARY.html`: High-level parameter summary and task division report.

### How to Run

You can execute the entire pipeline (installing dependencies, training the tokenizer, training the model, and evaluating) by running the following commands in your terminal:

```bash
# 1. Install dependencies
pip install torch numpy tqdm

# 2. Train the Tokenizer (Generates bpe_merges.json)
python train_bpe.py

# 3. Train the Model (Exactly 2,000 steps)
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt

# 4. Evaluate the Checkpoint (Calculates final bpb score)
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```
