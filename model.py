"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(text):
    unique_chars = list(set(text))
    srted = "".join(sorted(unique_chars))
    return {char: i for i, char in enumerate(srted)}

# Step 2 - build_id_to_char
def build_id_to_char(char_to_id):
    return {i: char for char, i in char_to_id.items()}

# Step 3 - encode_text
import numpy as np

def encode_text(text, char_to_id):
    encoded = [char_to_id[char] for char in text]
    return np.asarray(encoded, dtype=np.int64)

# Step 4 - decode_ids
def decode_ids(ids, id_to_char):
    return "".join([id_to_char[i] for i in ids])

# Step 5 - make_batches
import numpy as np

def make_batches(data, batch_size, block_size, rng):
    ix = rng.integers(0, len(data) - block_size, size=batch_size)

    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64)

    return x, y

