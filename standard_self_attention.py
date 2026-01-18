'''

Standard Self-Attention with RoPE for Positional Embeddings

'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import math

# Optimization for TF32 tensor cores
torch.set_float32_matmul_precision('high')

##########################################################################################################
# MAKE HYPER PARAMETER CHANGES IN THIS BLOCK
##########################################################################################################
hparams = {
    'n_embd': 256,  # number of embedding dimensions 384/6=64 for each head... must be divisible by n_head
    'n_head': 4,    # number of heads
    'n_layer': 4,   # number of layers
    'block_size': 256, # sequence length -- the max length for predictions
    'dropout_percentage': 0.5,
    'ffn_fanout': 4, # Default is 4
    'n_ffn': 1,
    'random_seed': 42,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'max_iters': 100_000,
    'eval_interval': 1000,        # Should be a reasonable fraction of max_iters
    'estimate_loss_iters': 1000,  # Should be a reasonable multiple of vocab_size
    'input': 'DATA/dickens.txt'
}
##########################################################################################################
##########################################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using: {device}')

# Report on hyper parameter values
# -------------------------
print("Hyper-parameters")
print("================")
print(f'n_embd = {hparams["n_embd"]}')
print(f'n_head = {hparams["n_head"]}')
print(f'n_layer = {hparams["n_layer"]}')
print(f'block_size = {hparams["block_size"]}')
print(f'dropout_percentage = {hparams["dropout_percentage"]}')
print(f'learning_rate = {hparams["learning_rate"]}')
print(f'batch_size = {hparams["batch_size"]}')
print(f'max_iters = {hparams["max_iters"]}')
print(f'eval_interval = {hparams["eval_interval"]}')
print(f'estimate_loss_iters = {hparams["estimate_loss_iters"]}')
print(f'ffn_fanout = {hparams["ffn_fanout"]}')
print(f'n_ffn = {hparams["n_ffn"]}')
print(f'random_seed = {hparams["random_seed"]}')
print()
# -------------------------

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

# For reproducibility
torch.manual_seed(hparams['random_seed'])

assert n_embd//n_head == n_embd/n_head, "Error: embedding dimension (n_embd) must be divisible by the number of heads (n_head)"

# Read data
with open(input, "r", encoding="utf-8") as file:
    data = file.read()
print(f'Total length: {len(data)/1000000:.2f} M characters')

# All unique characters in the dataset
chars = sorted(list(set(data)))
vocab_size = len(chars)

# Report on raw data stats
# -------------------------
print("Data Source + Info")
print("==================")
print(f'Source data: {input}')
print(f'Total length: {len(data)/1000000} M characters')
print(f'Vocab size: {vocab_size}')
print(f'Entropy: {-torch.log(torch.tensor(1/vocab_size)):.1f}')
print()
# -------------------------

# create a mapping of unique characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]  # encoder: convert a string to a list of integers
decode = lambda l: "".join([itos[i] for i in l])  # decoder: convert a list of integers to a string

# Split dataset into training and validation sets
data_ = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9 * len(data_))  # 90% of the data for training and 10% for validation
train_data, valid_data = data_[:n], data_[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of input x and targets y
    data = train_data if split == "train" else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))        # starting index for each sequence
    x = torch.stack([data[i : i + block_size] for i in ix])          # input data
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # target data
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(estimate_loss_iters)
        for k in range(estimate_loss_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_percentage)
        
    def apply_rotary_emb_static(self, x, seq_len):
        """Apply Rotary Position Embeddings to tensor x (B, T, C)"""
        B, T, C = x.shape

        # Get positions
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, T)

        # Compute frequencies (theta) for RoPE
        freqs = 10000 ** (-torch.arange(0, C, 2, device=x.device) / C)  # (C/2)
        angles = positions[:, :, None] * freqs[None, None, :]  # (1, T, C/2)
        cos_angles = torch.cos(angles)  # (1, T, C/2)
        sin_angles = torch.sin(angles)  # (1, T, C/2)

        # Split x into pairs for rotation
        x1, x2 = x[..., 0::2], x[..., 1::2]  # (B, T, C/2)
        x_rot = torch.cat([
            x1 * cos_angles - x2 * sin_angles,  # Rotate x1
            x1 * sin_angles + x2 * cos_angles   # Rotate x2
        ], dim=-1)  # (B, T, C)
        return x_rot
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        
        # Apply RoPE to keys and queries
        k = self.apply_rotary_emb_static(k, T)
        q = self.apply_rotary_emb_static(q, T)        # compute the attention scores ("affinities")
        
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # mask out the lower half of the matrix
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, H)
        out = wei @ v  # (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    """a multi-head attention module"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_percentage)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity"""

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
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: number of embedding dimensions, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffs = nn.ModuleList([FeedForward(n_embd) for _ in range(n_ffn)])
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = self.ln2(x)
        ff_out = torch.mean(torch.stack([ff(x) for ff in self.ffs]), dim=0)
        x = x + ff_out
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.scale = 1.0/math.sqrt(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx)  # (Batch, Time, Channels)
        x = self.blocks(x)
        logits = x @ self.token_embedding_table.weight.T * self.scale  # (Batch, Time, Vocab Size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # becomes (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # becomes (B, T+1)
        return idx

# Build model
model = GPTLanguageModel()
model = model.to(device)  # Move the model to the MPS device
model = torch.compile(model)

# print the number of parameters in the model

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

# Loop
start_time = datetime.now()
print(f"Datetime start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total iterations: {max_iters}")
iteration = []
val_mean_cross_entropy_loss = []
train_mean_cross_entropy_loss = []
for iter in range(1,max_iters+1):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter} | train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}", end=' ')
        elapsed_time, remaining_time, next_iter = remaining(datetime.now(), start_time, iter)
        print(f'| elapsed = {elapsed_time}, next = {next_iter}, remain = {remaining_time}')
        iteration.append(iter)
        train_mean_cross_entropy_loss.append(losses['train'])
        val_mean_cross_entropy_loss.append(losses['val'])
                    
    # sample a batch of data
    xb, yb = get_batch("train")

    # Check if autocast is implemented on the GPU. Autocast
    # automatically performs safe computations with the bfloat16,
    # applying it to layers not affected by lower precision, and using
    # float32 for the other layers that need it.
    #
    # NB: on the RTX 4090 GPU, autocast applied to this model achieves
    # a 25% reduction in GPU memory and a 26% speed up in computation
    # time without any discernable difference in training and
    # validation loss
    if torch.amp.is_autocast_available(device_type='cuda'):
        with torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
            logits, loss = model(xb, yb)
            loss.backward()
            
    # Don't use autocast if it isn't available
    else:
        logits, loss = model(xb, yb)
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# generate from the model using single char (\n) as input
prompt = '\n'

# Using the prompt, generate from the model
import time
start_time = time.time()
max_new_tokens = 1000
with torch.no_grad():
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    result = decode(model.generate(idx=idx, max_new_tokens=max_new_tokens)[0].tolist())

print()
print("-" * 80)
print(f'PROMPT:\n{prompt}')
print('OUTPUT:\n', result)
print()

end_time = time.time()
time_per_token_generated = (end_time - start_time)/10000
print(f"Time per token generated: {time_per_token_generated:.4f} seconds")

