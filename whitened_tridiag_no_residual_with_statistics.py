import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import math
import multiprocessing

import warnings
warnings.filterwarnings("ignore", message=".*TF32 behavior.*")

# Optimization for TF32 tensor cores
torch.set_float32_matmul_precision('high')

# Hyperparameters
hparams = {
    'n_embd': 256,
    'n_head': 2,
    'n_layer': 2,
    'block_size': 256,
    'dropout_percentage': 0.5,
    'ffn_fanout': 4,
    'n_ffn': 1,
    'random_seed': 42,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'max_iters': 100_000,
    'eval_interval': 1000,
    'estimate_loss_iters': 1000,
    'input': '../DATA/dickens.txt',
    'stats_file': '2x2_test_stats_tridiag_whiten_no_residual.txt'
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using: {device}')

# Print hyperparameters
print("Hyper-parameters")
print("================")
for key, value in hparams.items():
    print(f'{key} = {value}')
print()

# Extract hyperparameters
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
block_size = hparams["block_size"]
dropout_percentage = hparams["dropout_percentage"]
learning_rate = hparams["learning_rate"]
batch_size = hparams["batch_size"]
max_iters = hparams["max_iters"]
eval_interval = hparams["eval_interval"]
estimate_loss_iters = hparams["estimate_loss_iters"]
ffn_fanout = hparams["ffn_fanout"]
n_ffn = hparams["n_ffn"]
random_seed = hparams["random_seed"]
input = hparams["input"]
stats_file = hparams["stats_file"]

# For reproducibility
torch.manual_seed(hparams['random_seed'])

assert n_embd//n_head == n_embd/n_head, "Error: embedding dimension (n_embd) must be divisible by the number of heads (n_head)"

# Read and inspect data
with open(input, "r", encoding="utf-8") as file:
    data = file.read()
print(f'Total length: {len(data)/1000000:.2f} M characters')

# Create vocabulary
chars = sorted(list(set(data)))
vocab_size = len(chars)

print("Data Source + Info")
print("==================")
print(f'Source data: {input}')
print(f'Total length: {len(data)/1000000} M characters')
print(f'Vocab size: {vocab_size}')
print(f'Entropy: {-torch.log(torch.tensor(1/vocab_size)):.1f}')
print()

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

# Split dataset
data_ = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9 * len(data_))
train_data, valid_data = data_[:n], data_[n:]

# Data loading
def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(estimate_loss_iters)
        for k in range(estimate_loss_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y, compute_whiteness=False)  # Disable whiteness computation
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def compute_sample_covariance(tensor):
    B, T, C = tensor.shape
    tensor = tensor - torch.mean(tensor, dim=1, keepdim=True)
    cov = torch.matmul(tensor.transpose(1, 2), tensor) / (T - 1)
    return cov

@torch.no_grad()
def measure_whiteness(cov):
    B, T, _ = cov.shape
    diag_mask = torch.eye(T, device=cov.device, dtype=torch.bool).unsqueeze(0).expand(B, T, T)
    off_diag_mask = ~diag_mask
    off_diag_mean = torch.mean(torch.abs(cov[off_diag_mask]).reshape(B, -1), dim=-1)
    diag_mean = torch.mean(torch.abs(cov[diag_mask]).reshape(B, -1), dim=-1)
    diag_mean = torch.where(diag_mean == 0, torch.ones_like(diag_mean), diag_mean)
    ratio = off_diag_mean / diag_mean
    return torch.mean(ratio)

@torch.no_grad()
def estimate_ar1_stationarity(x):
    """
    Estimates if sequences in x (B x T x C) are from a stationary AR(1) process.
    Computes first-order C x C sample cross-covariance matrices across batch dimension
    and measures their variation along the sequence dimension T.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, T, C)
    
    Returns:
        dict: Contains mean covariance matrix and variance of covariance matrices along T
    """
    B, T, C = x.shape
    
    # Center the data by subtracting the mean across the batch for each time step and channel
    x_centered = x - x.mean(dim=0, keepdim=True)
    
    # Initialize tensor to store covariance matrices for each time step
    cov_matrices = torch.zeros(T-1, C, C, device=x.device)
    
    # Compute sample cross-covariance matrices for lag 1
    for t in range(T-1):
        # x_t: (B, C), x_{t+1}: (B, C)
        x_t = x_centered[:, t, :]      # Shape: (B, C)
        x_t_next = x_centered[:, t+1, :]  # Shape: (B, C)
        
        # Compute sample covariance: (x_t^T x_{t+1}) / (B-1)
        cov_matrices[t] = torch.matmul(x_t.transpose(-1, -2), x_t_next) / (B - 1)
    
    # Compute mean covariance matrix across time steps
    mean_cov_matrix = cov_matrices.mean(dim=0)  # Shape: (C, C)
    
    # Compute variation of covariance matrices along T dimension
    # Frobenius norm of difference from mean for each covariance matrix
    cov_diff = cov_matrices - mean_cov_matrix
    cov_variation = torch.norm(cov_diff, p='fro', dim=(-2, -1))  # Shape: (T-1,)
    
    # Average variation across time steps
    mean_variation = cov_variation.mean()
    
    return {
        'mean_cov_matrix': mean_cov_matrix,
        'cov_variation': cov_variation,
        'mean_variation': mean_variation
    }

@torch.no_grad()
def evaluate_whiteness(x, w):
    cov_x = compute_sample_covariance(x)
    cov_w = compute_sample_covariance(w)
    whiteness_x = measure_whiteness(cov_x)
    whiteness_w = measure_whiteness(cov_w)
    is_whiter = whiteness_w < whiteness_x
    ar1_x = estimate_ar1_stationarity(x)
    return {
        'whiteness_x': whiteness_x,
        'whiteness_w': whiteness_w,
        'is_w_whiter': is_whiter,
        'ar1_x': ar1_x['mean_variation']
    }

class BlockTridiagonalWhitening(nn.Module):
    def __init__(self):
        super().__init__()
        self.V_0 = nn.Parameter(torch.tril(torch.randn(n_embd, n_embd)*.01))
        self.V_1 = nn.Parameter(torch.randn(n_embd, n_embd)*.01)

    def forward(self, x: torch.Tensor, compute_whiteness: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        x = x - x.mean(dim=-1, keepdim=True)
        w = torch.zeros_like(x)
        w[:, 0, :] = torch.matmul(x[:, 0, :], self.V_0)
        for i in range(1, T):
            prev_term = torch.matmul(w[:, i-1, :], self.V_1)
            diff = x[:, i, :] - prev_term
            w[:, i, :] = torch.matmul(diff, self.V_0)
        if compute_whiteness:
            results = evaluate_whiteness(x, w)
            return w, results
        return w, None

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_percentage)

    def apply_rotary_emb_static(self, x, seq_len):
        B, T, C = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        freqs = 10000 ** (-torch.arange(0, C, 2, device=x.device) / C)
        angles = positions[:, :, None] * freqs[None, None, :]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rot = torch.cat([
            x1 * cos_angles - x2 * sin_angles,
            x1 * sin_angles + x2 * cos_angles
        ], dim=-1)
        return x_rot

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        k = self.apply_rotary_emb_static(k, T)
        q = self.apply_rotary_emb_static(q, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ x
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_head*n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_percentage)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffn_fanout * n_embd, bias=True),
            nn.GELU(),
            nn.Linear(ffn_fanout * n_embd, n_embd, bias=True),
            nn.Dropout(dropout_percentage),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffs = nn.ModuleList([FeedForward(n_embd) for _ in range(n_ffn)])
        self.ln = nn.LayerNorm(n_embd)
        self.whitener = BlockTridiagonalWhitening()

    def forward(self, x, compute_whiteness=True):
        x, whitening_results = self.whitener(x, compute_whiteness)
        x = x + self.sa(x)
        x = self.ln(x)
        ff_out = torch.mean(torch.stack([ff(x) for ff in self.ffs]), dim=0)
        x = x + ff_out
        return x, whitening_results

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.scale = 1.0/math.sqrt(n_embd)

    def forward(self, idx, targets=None, compute_whiteness=True):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        whitening_results = []
        for block in self.blocks:
            x, block_results = block(x, compute_whiteness)
            if block_results is not None:
                whitening_results.append(block_results)
        logits = x @ self.token_embedding_table.weight.T * self.scale  # (Batch, Time, Vocab Size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, whitening_results

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond, compute_whiteness=False)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Build model
model = GPTLanguageModel()
model = model.to(device)
model = torch.compile(model)

print("Model Size")
print("==========")
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print()

# Instantiate optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Time tracking
def remaining(now, start, iter):
    delta = now - start
    total_seconds = delta.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    time_per_iter = total_seconds/iter
    remaining_seconds = (max_iters - iter) * time_per_iter
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    remaining = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    next_iter = (now + timedelta(seconds=time_per_iter*eval_interval)).strftime("%H:%M:%S")
    return elapsed, remaining, next_iter

# Training loop
start_time = datetime.now()
print(f"Datetime start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total iterations: {max_iters}")
iteration = []
val_mean_cross_entropy_loss = []
train_mean_cross_entropy_loss = []
for iter in range(1, max_iters+1):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter} | train_loss: {losses['train'].item():.4f}, val_loss: {losses['val'].item():.4f}", end=' ')
        elapsed_time, remaining_time, next_iter = remaining(datetime.now(), start_time, iter)
        print(f'| elapsed = {elapsed_time}, next = {next_iter}, remain = {remaining_time}')
        iteration.append(iter)
        train_mean_cross_entropy_loss.append(losses['train'].item())
        val_mean_cross_entropy_loss.append(losses['val'].item())

    xb, yb = get_batch("train")
    compute_whiteness = (iter % estimate_loss_iters == 0)  # Compute only every estimate_loss_iters
    if torch.amp.is_autocast_available(device_type='cuda'):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss, whitening_results = model(xb, yb, compute_whiteness=compute_whiteness)
            loss.backward()
    else:
        logits, loss, whitening_results = model(xb, yb, compute_whiteness=compute_whiteness)
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Write whitening results for each block every estimate_loss_iters
    if compute_whiteness and whitening_results:
        with open(stats_file, 'a') as f:
            for i, results in enumerate(whitening_results):
                f.write(f"Iter {iter} | Block {i+1} | x: {results['whiteness_x'].item():.4f} | w: {results['whiteness_w'].item():.4f} | r: {results['whiteness_w'].item()/results['whiteness_x'].item():.4f} | ar1: {results['ar1_x'].item():.4f}\n")
                
# Generate from model
prompt = '\n'
with torch.no_grad():
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    result = decode(model.generate(idx=idx, max_new_tokens=1000)[0].tolist())

print()
print("-" * 80)
print(f'PROMPT:\n{prompt}')
print('OUTPUT:\n', result)
print()

# Plot validation loss
plt.figure(figsize=(12, 8))
x = iteration
y1 = train_mean_cross_entropy_loss
y2 = val_mean_cross_entropy_loss
plt.plot(x, y1, marker='o', linestyle='-', label='train')
plt.plot(x, y2, marker='o', linestyle='-', label='val')
plt.title('Training Performance', fontsize=24)
plt.xlabel('Training Iterations', fontsize=19)
plt.ylabel('Mean Cross-Entropy Loss', fontsize=19)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=15)
plt.grid(True)
plt.ion()
plt.show()
