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

# Step 6 - token_embedding_lookup
def token_embedding_lookup(token_ids, embedding_table):
    d_model = embedding_table.shape[1]
    b, t = token_ids.shape
    embeddings = embedding_table[token_ids]

    return embeddings.reshape(b, t, d_model)

# Step 7 - add_positional_embeddings
import numpy as np

def add_positional_embeddings(token_embeds, pos_embedding_table):
    # TODO: add learned positional embeddings (sliced to length T) to token embeddings
    t = token_embeds.shape[1]
    sliced_embedding_table = pos_embedding_table[:t]
    return token_embeds + sliced_embedding_table

# Step 8 - linear_projection
import numpy as np

def linear_projection(x, weight, bias):
    projection = x @ weight

    if bias is not None:
        projection = projection + bias

    return projection

# Step 9 - compute_attention_scores
def compute_attention_scores(q, k):
    return q @ np.matrix_transpose(k)

