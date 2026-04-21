"""
nanoGPT — Step 1: Data Loading
==============================
Goal: Read a text file, build a vocabulary, encode the text as integers,
      and split into train/val sets.

Instructions: Fill in every line marked with TODO. Run the file when you're
done — if everything prints correctly, you're ready for Step 2.
"""

import torch

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
block_size = 8    # context length (we'll increase this later)
batch_size = 4    # number of independent sequences per batch

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

    return x, y

# Test it
xb, yb = get_batch('train')
print(f"Input shape:  {xb.shape}")   # should be [4, 8]
print(f"Target shape: {yb.shape}")   # should be [4, 8]
print(f"\nFirst input sequence:  {xb[0].tolist()}")
print(f"First target sequence: {yb[0].tolist()}")

import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # TODO: create an embedding table of shape (vocab_size, vocab_size)
        # Each token looks up a row of logits predicting the next token
        # Hint: nn.Embedding(vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        # idx is shape (batch_size, block_size) — the input tokens
        # TODO: pass idx through the embedding table to get logits
        # Result shape: (batch_size, block_size, vocab_size)
        logits =  self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # PyTorch's cross_entropy expects shape (N, C) not (B, T, C)
            # so we need to reshape
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) — current context
        for _ in range(max_new_tokens):
            # TODO: get logits from forward pass (ignore loss)
            logits, loss = self(idx)

            # only look at the last time step
            logits = logits[:, -1, :]  # (B, C)

            # TODO: convert logits to probabilities
            # Hint: F.softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            # TODO: sample one token from the probability distribution
            # Hint: torch.multinomial(probs, num_samples=1)
            idx_next = torch.multinomial(probs,num_samples=1)

            # TODO: append the new token to the running sequence
            # Hint: torch.cat((idx, idx_next), dim=1)
            idx = torch.cat((idx, idx_next), dim=-1)

        return idx

# Create the model
model = BigramLanguageModel(vocab_size)

# Test generation BEFORE training (should be total garbage)
start = torch.zeros((1, 1), dtype=torch.long)  # start with token 0
print("Before training:")
print(decode(model.generate(start, max_new_tokens=100)[0].tolist()))

# --------------------------------------------------------------------------
# Step 4: Training loop
# --------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10000):

    # TODO: grab a batch of training data
    # Hint: use get_batch('train')
    xb, yb = get_batch('train')

    # TODO: forward pass — get logits and loss
    logits, loss = model(xb,yb)

    # These three lines are always the same — the core of training
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print loss every 1000 steps
    if step % 1000 == 0:
        print(f"Step {step:5d} | Loss: {loss.item():.4f}")

# Generate text AFTER training
print("\nAfter training:")
start = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))