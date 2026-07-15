# RUNLOG

## Run 1: Baseline
I started from the default unoptimized setup and observed high loss together with slow convergence. The constant learning rate and standard LayerNorm did not make efficient use of the limited 2000-step budget. This established the reference point for later improvements.

## Run 2: Architecture Upgrade
I replaced LayerNorm with RMSNorm, swapped GELU for SwiGLU, and tied the token embedding and output weights. The hypothesis was that RMSNorm would reduce CPU overhead, SwiGLU would improve non-linearity, and tied weights would save parameters that could be reinvested into depth. This enabled a 4-layer model that stayed within the 2,000,000 parameter limit and reduced validation loss.

## Run 3: Optimizer Upgrade
I switched from standard Adam to AdamW, added cosine annealing, introduced 200 warmup steps, and enabled gradient clipping. The hypothesis was that warmup would prevent early instability, cosine decay would let the model settle into a sharper minimum, and clipping would protect the run under the hard 2000-step cap. The result was a large drop in training loss and more stable optimization.

## Run 4: Tokenizer Upgrade
I replaced the byte tokenizer with a 512-vocabulary BPE tokenizer trained on the provided corpus. The hypothesis was that the byte tokenizer wastes context on multi-byte Hindi characters, while BPE compresses the sequence and exposes longer-range dependencies to the model. This produced the best bits-per-byte score by improving effective context utilization.