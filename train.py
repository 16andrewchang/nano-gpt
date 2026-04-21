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