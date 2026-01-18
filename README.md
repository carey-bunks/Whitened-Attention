# Whitened Self-Attention (WSA)

This repository contains two implementations of Whitened Self-Attention (WSA), a theoretically motivated enhancement for Transformer architectures that accounts for inter-token correlations.  The first implementation is based on the block tridiagonal approximation to the autocovariance matrix, and the second is block pentadiagonal.  For comparisons, there is also a separate standard attention implementation (no whitening).

This code is based on the following paper:
_Whitened Self-Attention_, anonymous authors, under review as a conference paper at ICLR 2026. Read the full paper on [OpenReview](https://openreview.net/pdf?id=XQ0VTUIhEJ)

## Overview

Standard self-attention treats context vectors as mutually independent, which introduces information duplication when tokens are correlated. Whitened Self-Attention (WSA) addresses this by applying a computationally feasible whitening filter that decorrelates input sequences, producing an optimal, minimum variance estimator.

### Key Results
* **Efficiency:** Achieves the same Mean Cross-Entropy (MCE) loss as standard attention in 37x fewer training iterations.
* **Speed:** Reduces total training time by up to 91% after hyperparameter optimization.
* **Performance:** Reduces perplexity by 19.3% compared to standard GPT architectures of equivalent capacity.

## Architecture

The implementation is based on a GPT-style decoder Transformer with the following features:
- **Whitening Filter:** A learned recursion block implementing the transformation:
  w[0] = inv(L) * x[0]
  w[i] = inv(L) * (x[i] - M * w[i-1])
  where L and M are learned steady-state covariance matrices.
- **Attention:** A modified self-attention mechanism that operates on whitened vectors, effectively absorbing the standard "Value" (V) matrix into the whitening process.
- **Positional Encoding:** Rotary Positional Embeddings (RoPE).

## Installation

```bash
git clone [https://github.com/](https://github.com/)[your-username]/whitened-attention.git
cd whitened-attention
pip install -r requirements.txt
