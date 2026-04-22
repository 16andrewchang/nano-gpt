"""
nanoGPT — Step 1: Data Loading
==============================
Goal: Read a text file, build a vocabulary, encode the text as integers,
      and split into train/val sets.

Instructions: Fill in every line marked with TODO. Run the file when you're
done — if everything prints correctly, you're ready for Step 2.
"""

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# --------------------------------------------------------------------------
# 1. Read the dataset
# --------------------------------------------------------------------------
# TODO: open 'input.txt' and read the entire contents into a string called `text`
# Hint: use Python's built-in open() with encoding='utf-8'

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters in dataset: {len(text)}")
print(f"First 200 characters:\n{text[:200]}\n")

# --------------------------------------------------------------------------
# 2. Build the vocabulary
# --------------------------------------------------------------------------
# The vocabulary is just the set of all unique characters in the text.

# TODO: create a sorted list of all unique characters in `text`
# Hint: sorted(list(set(...)))

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}")
print(f"Characters: {''.join(chars)}\n")

# --------------------------------------------------------------------------
# 3. Build the encoder and decoder
# --------------------------------------------------------------------------
# We need two mappings:
#   stoi  (string-to-integer):  'a' -> 0, 'b' -> 1, ...
#   itos  (integer-to-string):  0 -> 'a', 1 -> 'b', ...
#
# Then two helper functions:
#   encode(string)   -> list of integers
#   decode(integers) -> string

# TODO: build stoi — a dict mapping each character to its index in `chars`
# Hint: { ch: i for i, ch in enumerate(chars) }

stoi = {ch: i for i, ch in enumerate(chars)}

# TODO: build itos — a dict mapping each index back to its character

itos = {i: ch for i, ch in enumerate(chars)}

# TODO: write encode() — takes a string, returns a list of ints
# Hint: use a list comprehension with stoi

def encode(s): 
    res = []
    for ch in s:
        res.append(stoi[ch])

    return res

# TODO: write decode() — takes a list of ints, returns a string
# Hint: use ''.join() with itos

def decode(l):
    return ''.join([itos[i] for i in l ])

# Quick sanity check — this should print True
test_str = "hello"
print(f"Encode/decode sanity check: {decode(encode(test_str)) == test_str}")  # should be True
print(f"  '{test_str}' encodes to {encode(test_str)}")
print(f"  which decodes back to '{decode(encode(test_str))}'\n")

# --------------------------------------------------------------------------
# 4. Encode the full dataset and make it a torch tensor
# --------------------------------------------------------------------------
# TODO: encode the entire `text` and wrap it in a torch.long tensor
# Hint: torch.tensor(encode(...), dtype=torch.long)

data = torch.tensor(encode(text),dtype = torch.long)

print(f"Data tensor shape: {data.shape}")
print(f"Data dtype:        {data.dtype}\n")

# --------------------------------------------------------------------------
# 5. Train/val split
# --------------------------------------------------------------------------
# Use the first 90% for training, the last 10% for validation.

# TODO: compute the split index (integer, 90% of total length)

n = int(len(data)*0.9)

# TODO: slice `data` into train_data and val_data

train_data = data[:n]
val_data   = data[n:]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}\n")

# --------------------------------------------------------------------------
# 6. Done! If you see reasonable numbers above, Step 1 is complete.
# --------------------------------------------------------------------------
# Expected output for Tiny Shakespeare:
#   - Total characters: ~1,115,394
#   - Vocab size: 65
#   - Train tokens: ~1,003,854
#   - Val tokens:   ~111,540
#
# Next step: we'll write a function that grabs random mini-batches
# from this data. But that's for Step 2 — take a break first.

# --------------------------------------------------------------------------
# Step 2: Batching
# --------------------------------------------------------------------------
block_size = 64    # context length (we'll increase this later)
batch_size = 32   # number of independent sequences per batch
n_embd = 128
def get_batch(split):
    # TODO: pick the right dataset based on `split`
    # Hint: if split == 'train' use train_data, else val_data
    d = train_data if split == 'train' else val_data

    # TODO: generate `batch_size` random starting positions
    # Each position must leave room for block_size + 1 tokens
    # Hint: torch.randint(0, len(d) - block_size, (batch_size,))
    ix = torch.randint(0,len(d)-block_size,(batch_size,))

    # TODO: for each starting position, grab a chunk of length block_size -> that's x
    # and a chunk shifted by 1 -> that's y (the targets)
    # Hint: torch.stack([d[i:i+block_size] for i in ix])
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Test it
xb, yb = get_batch('train')
print(f"Input shape:  {xb.shape}")   # should be [4, 8]
print(f"Target shape: {yb.shape}")   # should be [4, 8]
print(f"\nFirst input sequence:  {xb[0].tolist()}")
print(f"First target sequence: {yb[0].tolist()}")

import torch.nn as nn
from torch.nn import functional as F
# --------------------------------------------------------------------------
# Step 5: Self-Attention
# --------------------------------------------------------------------------

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        # TODO: create three linear projections (no bias) for key, query, value
        # Each takes n_embd as input and outputs head_size
        # Hint: nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # This creates the causal mask — tokens can't look at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # TODO: compute key and query for all tokens
        k = self.key(x) # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores ("affinities")
        # TODO: dot product of queries and keys, scaled by sqrt(head_size)
        # Hint: q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)

        # Apply causal mask — fill future positions with -inf so softmax gives 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # TODO: apply softmax to get attention weights
        wei = F.softmax(wei,dim = -1) # (B, T, T)

        # TODO: compute value vectors and multiply by attention weights
        v = self.value(x)    # (B, T, head_size)
        out = wei @ v # (B, T, head_size)

        return out
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # TODO: create a list of `num_heads` Head modules
        # Hint: nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # TODO: a linear projection to combine the outputs
        # All heads concatenate to num_heads * head_size, project back to n_embd
        # Hint: nn.Linear(num_heads * head_size, n_embd)
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # TODO: run each head on x, concatenate their outputs along the last dimension
        # Hint: torch.cat([h(x) for h in self.heads], dim=-1)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # TODO: pass through the projection layer
        out = self.proj(out)

        return out
class FeedForward(nn.Module):
    """A simple two-layer network applied to each token independently."""

    def __init__(self, n_embd):
        super().__init__()
        # TODO: create a sequential network:
        #   1. Linear from n_embd to 4 * n_embd (expand)
        #   2. ReLU activation
        #   3. Linear from 4 * n_embd back to n_embd (compress)
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
        )

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Token embeddings now map to n_embd, not vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # TODO: add position embeddings — the model needs to know WHERE each token is
        # Same idea: an embedding table of shape (block_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        n_layer = 4
        n_head = 4
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        # TODO: final linear layer to go from n_embd back to vocab_size for predictions
        # Hint: nn.Linear(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # TODO: get token embeddings — shape (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)

        # TODO: get position embeddings for positions 0..T-1
        # Hint: torch.arange(T) gives you [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # TODO: add token and position embeddings together
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # TODO: pass through attention head
        x = self.blocks(x)  # (B, T, n_embd)

        # TODO: pass through final linear layer to get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop idx to last block_size tokens so position embeddings don't go out of bounds
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Create the model
model = BigramLanguageModel(vocab_size)
model = model.to(device)
# Test generation BEFORE training (should be total garbage)
start = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with token 0
print("Before training:")
print(decode(model.generate(start, max_new_tokens=100)[0].tolist()))
@torch.no_grad()
def estimate_loss(eval_iters=200):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# --------------------------------------------------------------------------
# Step 4: Training loop
# --------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(15000):

    if step % 500 == 0:
        losses = estimate_loss()
        print(f"Step {step:5d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text AFTER training
print("\nAfter training:")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(start, max_new_tokens=1000)[0].tolist())
print(generated_text)

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)