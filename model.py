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

# Step 10 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, head_dim):
    return scores / np.sqrt(head_dim)

# Step 11 - apply_causal_mask
import numpy as np

def apply_causal_mask(scores):
    causal_mask = np.tril(np.ones_like(scores, dtype=bool))

    return np.where(causal_mask, scores, -np.inf)

# Step 12 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    masked_scores = masked_scores - np.max(masked_scores, axis=-1, keepdims=True)
    return np.exp(masked_scores) / np.sum(np.exp(masked_scores), axis=-1, keepdims=True)

# Step 13 - attention_context
import numpy as np

def attention_context(attn_weights, v):
    return attn_weights @ v

# Step 14 - split_heads
import numpy as np

def split_heads(x, num_heads):
    """Reshape (B, T, d_model) into (B, num_heads, T, head_dim)."""
    B, T, d_model = x.shape
    head_dim = d_model // num_heads

    x = x.reshape(B, T, num_heads, head_dim)
    x = np.swapaxes(x, 1, 2)

    return x

# Step 15 - merge_heads
import numpy as np

def merge_heads(x):
    B, num_heads, T, head_dim = x.shape
    d_model = num_heads * head_dim
    x = np.swapaxes(x, 1, 2)

    return x.reshape(B, T, d_model)

# Step 16 - project_qkv
def project_qkv(x, w_q, b_q, w_k, b_k, w_v, b_v):
    Q = linear_projection(x, w_q, b_q)
    K = linear_projection(x, w_k, b_k)
    V = linear_projection(x, w_v, b_v)

    return Q, K, V

# Step 17 - multi_head_scaled_dot_product_attention
import numpy as np

def multi_head_scaled_dot_product_attention(q, k, v, num_heads):
    """Run masked scaled dot-product attention in parallel across num_heads.

    q, k, v: arrays of shape (B, T, d_model).
    Returns: per-head context of shape (B, num_heads, T, head_dim).
    """
    B, T, d_model = q.shape
    head_dim = d_model // num_heads

    qh = split_heads(q, num_heads)
    kh = split_heads(k, num_heads)
    vh = split_heads(v, num_heads)

    attention_scores = compute_attention_scores(qh, kh)

    scaled_scores = scale_attention_scores(attention_scores, head_dim)
    masked_scores = apply_causal_mask(scaled_scores)
    attention_weights = softmax_attention_weights(masked_scores)

    return attention_context(attention_weights, vh)

# Step 18 - merge_and_output_project
import numpy as np

def merge_and_output_project(head_context, w_o, b_o):
    context = merge_heads(head_context)
    return linear_projection(context, w_o, b_o)

# Step 19 - masked_multi_head_self_attention
def masked_multi_head_self_attention(x, params, num_heads):
    Q, K, V = project_qkv(
        x, params['w_q'], params['b_q'], params['w_k'], params['b_k'], params['w_v'], params['b_v']
        )
    
    head_context = multi_head_scaled_dot_product_attention(Q, K, V, num_heads)
    context = merge_and_output_project(head_context, params['w_o'], params['b_o'])

    return context

# Step 20 - gelu_activation
import numpy as np

def gelu_activation(x):
    """Apply tanh-approximate GELU elementwise. Return array of same shape."""
    x_cubed_constant = 0.044715 * x**3
    u = np.sqrt(2 / np.pi) * (x + x_cubed_constant)
    return (x / 2) * (1 + np.tanh(u))

# Step 21 - ffn_first_layer
import numpy as np

def ffn_first_layer(x, w1, b1):
    x_projection = linear_projection(x, w1, b1)
    x_activated = gelu_activation(x_projection)
    return x_activated

# Step 22 - ffn_second_layer
import numpy as np

def ffn_second_layer(h, w2, b2):
    return linear_projection(h, w2, b2)

