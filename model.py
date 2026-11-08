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

# Step 23 - position_wise_feed_forward
import numpy as np

def position_wise_feed_forward(x, params):
    ff1 = ffn_first_layer(x, params['w1'], params['b1'])
    ff2 = ffn_second_layer(ff1, params['w2'], params['b2'])
    return ff2

# Step 24 - layernorm_stats
def layernorm_stats(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return mean, variance

# Step 25 - layer_norm
def layer_norm(x, gamma, beta, eps=1e-5):
    mean, variance = layernorm_stats(x, eps)
    x_hat = (x - mean) / np.sqrt(variance + eps)
    
    return gamma * x_hat + beta

# Step 26 - pre_norm_residual_sublayer
import numpy as np

def pre_norm_residual_sublayer(x, gamma, beta, sublayer_fn):
    return x + sublayer_fn(layer_norm(x, gamma, beta))

# Step 27 - transformer_block
def transformer_block(x, params, num_heads):
    """One transformer block: pre-norm attention sublayer then pre-norm FFN sublayer."""
    x = pre_norm_residual_sublayer(
        x,
        params['ln1_gamma'],
        params['ln1_beta'],
        lambda normed_x: masked_multi_head_self_attention(
            normed_x, params['attn'], num_heads
        ))

    x = pre_norm_residual_sublayer(
        x,
        params['ln2_gamma'],
        params['ln2_beta'],
        lambda normed_x: position_wise_feed_forward(
            normed_x, params['ffn']
        ))
    return x

# Step 28 - gpt_backbone
def gpt_backbone(x, blocks_params, num_heads):
    # TODO: run x through each transformer block in blocks_params sequentially and return the final tensor.
    for params in blocks_params:
        x = transformer_block(x, params, num_heads)

    return x

# Step 29 - project_to_vocab_logits
import numpy as np

def project_to_vocab_logits(hidden_states, w_out, b_out):
    return linear_projection(hidden_states, w_out, b_out)

# Step 30 - gpt_forward
def gpt_forward(token_ids, params, num_heads):
    embeddings = token_embedding_lookup(token_ids, params['token_embedding'])
    pos_emb = params['pos_embedding'] if 'pos_embedding' in params else params['positional_embedding']
    context_aware = add_positional_embeddings(embeddings, pos_emb)
    normed = layer_norm(context_aware, params['ln_f_gamma'], params['ln_f_beta'])
    hidden_states = gpt_backbone(normed, params['blocks'], num_heads)

    return project_to_vocab_logits(hidden_states, params['w_out'], params['b_out'])

# Step 31 - cross_entropy_language_modeling_loss
def cross_entropy_language_modeling_loss(logits, targets):
    shifted = logits - logits.max(axis=-1, keepdims=True)

    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    batch, seq = targets.shape
    selected = log_probs[
        np.arange(batch)[:, None],
        np.arange(seq)[None, :],
        targets,
    ]

    return float(-selected.mean())

# Step 32 - init_gpt_parameters
def init_gpt_parameters(vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers, seed=0):
    rng = np.random.default_rng(seed)
    std = 0.02
    params = {
        "token_embedding": rng.normal(0.0, std, size=(vocab_size, d_model)),
        "positional_embedding": rng.normal(
            0.0, std, size=(max_seq_len, d_model)
        ),
        "blocks": [],
        "ln_f_gamma": np.ones((d_model,)),
        "ln_f_beta": np.zeros((d_model,)),
        "w_out": rng.normal(0.0, std, size=(d_model, vocab_size)),
        "b_out": np.zeros((vocab_size,)),
    }
    for _ in range(num_layers):
        block = {
            "ln1_gamma": np.ones((d_model,)),
            "ln1_beta": np.zeros((d_model,)),
            "attn": {
                "w_q": rng.normal(0.0, std, size=(d_model, d_model)),
                "b_q": np.zeros((d_model,)),
                "w_k": rng.normal(0.0, std, size=(d_model, d_model)),
                "b_k": np.zeros((d_model,)),
                "w_v": rng.normal(0.0, std, size=(d_model, d_model)),
                "b_v": np.zeros((d_model,)),
                "w_o": rng.normal(0.0, std, size=(d_model, d_model)),
                "b_o": np.zeros((d_model,)),
            },
            "ln2_gamma": np.ones((d_model,)),
            "ln2_beta": np.zeros((d_model,)),
            "ffn": {
                "w1": rng.normal(0.0, std, size=(d_model, d_ff)),
                "b1": np.zeros((d_ff,)),
                "w2": rng.normal(0.0, std, size=(d_ff, d_model)),
                "b2": np.zeros((d_model,)),
            },
        }
        params["blocks"].append(block)
    return params

# Step 33 - collect_parameters
def collect_parameters(params):
    def append_nested(container, collected):
        for value in container.values():
            if isinstance(value, dict):
                append_nested(value, collected)
            else:
                collected.append(value)

    collected = []

    # Embeddings
    collected.append(params["token_embedding"])
    collected.append(params["positional_embedding"])

    # Transformer blocks
    for block in params["blocks"]:
        append_nested(block, collected)

    # Final layer norm
    collected.append(params["ln_f_gamma"])
    collected.append(params["ln_f_beta"])

    # Output projection
    collected.append(params["w_out"])
    collected.append(params["b_out"])

    return collected

# Step 34 - training_step
def training_step(params, input_ids, target_ids):
    eps = 1e-5
    num_heads = params['num_heads']

    def compute_loss(current_params):
        logits = gpt_forward(input_ids, current_params, num_heads)
        return cross_entropy_language_modeling_loss(logits, target_ids)

    loss = compute_loss(params)

    grads = {}

    for name, value in params.items():
        if not isinstance(value, np.ndarray):
            continue

        grad = np.zeros_like(value, dtype=np.float64)
        flat_param = value.reshape(-1)
        flat_grad = grad.reshape(-1)

        for i in range(flat_param.size):
            original = flat_param[i]

            flat_param[i] = original + eps
            loss_plus = compute_loss(params)

            flat_param[i] = original - eps
            loss_minus = compute_loss(params)

            flat_param[i] = original
            flat_grad[i] = (loss_plus - loss_minus) / (2.0 * eps)

        grads[name] = grad

    return loss, grads

# Step 35 - apply_optimizer_update
def apply_optimizer_update(params, grads, learning_rate):
    if isinstance(params, dict) and isinstance(grads, dict):
        for key, param in params.items():
            if key not in grads:
                continue
            grad = grads[key]
            if isinstance(param, (dict, list, tuple)):
                apply_optimizer_update(param, grad, learning_rate)
            elif isinstance(param, np.ndarray):
                param -= learning_rate * grad
            elif isinstance(param, (int, float, np.number)) and grad is not None:
                params[key] = param - learning_rate * grad
    elif isinstance(params, (list, tuple)) and isinstance(grads, (list, tuple)):
        for i, (param, grad) in enumerate(zip(params, grads)):
            if isinstance(param, (dict, list, tuple)):
                apply_optimizer_update(param, grad, learning_rate)
            elif isinstance(param, np.ndarray):
                param -= learning_rate * grad
            elif isinstance(param, (int, float, np.number)) and grad is not None and isinstance(params, list):
                params[i] = param - learning_rate * grad
    elif isinstance(params, np.ndarray) and isinstance(grads, np.ndarray):
        params -= learning_rate * grads
    return params

# Step 36 - run_training_loop
def run_training_loop(params, batches, num_steps, learning_rate):
    losses = []

    for i in range(num_steps):
        input_ids, target_ids = batches[i % len(batches)]
        loss, grads = training_step(params, input_ids, target_ids)
        losses.append(loss)
        params = apply_optimizer_update(params, grads, learning_rate)
    
    return losses

# Step 37 - last_position_logits
def last_position_logits(logits):
    return logits[:, -1, :]

# Step 38 - scale_logits_by_temperature
def scale_logits_by_temperature(logits, temperature):
    return logits / temperature

# Step 39 - top_k_filter_logits
def top_k_filter_logits(logits, k):
    vocab_size = logits.shape[-1]

    if k >= vocab_size:
        return logits

    kth_largest = np.partition(logits, -k, axis=-1)[:, -k, None]

    return np.where(
        logits >= kth_largest,
        logits,
        np.array(-np.inf, dtype=logits.dtype),
    )

# Step 40 - sample_next_token
def sample_next_token(filtered_logits, rng):
    filtered_logits = np.asarray(filtered_logits)
    batch_size, vocab_size = filtered_logits.shape

    finite_mask = np.isfinite(filtered_logits)
    if not np.all(finite_mask.any(axis=1)):
        raise ValueError("At least one row has no finite logits.")

    safe_logits = np.where(finite_mask, filtered_logits, -np.inf)
    max_logits = np.max(safe_logits, axis=1, keepdims=True)
    exp_logits = np.exp(safe_logits - max_logits)
    exp_logits = np.where(finite_mask, exp_logits, 0.0)

    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum(axis=1, keepdims=True)

    cum_probs = np.cumsum(probs, axis=1)
    cum_probs[:, -1] = 1.0  

    u = rng.random(batch_size)
    next_tokens = np.argmax(cum_probs >= u[:, None], axis=1)

    return next_tokens.astype(np.int64)

# Step 41 - generate_text
def generate_text(params, prompt, num_new_tokens, max_seq_len, vocab, id_to_char, temperature=1.0, top_k=None, seed=0):
    rng = np.random.default_rng(seed)
    num_heads = params['num_heads']
    
    token_ids = list(encode_text(prompt, vocab))
    
    for _ in range(num_new_tokens):
        context_ids = np.asarray([token_ids[-max_seq_len:]], dtype=np.int64)
        
        logits = gpt_forward(context_ids, params, num_heads)
        
        next_logits = last_position_logits(logits)
        
        scaled_logits = scale_logits_by_temperature(next_logits, temperature)
        if top_k is not None:
            filtered_logits = top_k_filter_logits(scaled_logits, top_k)
        else:
            filtered_logits = scaled_logits
            
        next_id = sample_next_token(filtered_logits, rng)
        token_ids.append(int(np.asarray(next_id).item()))
        
    return decode_ids(token_ids, id_to_char)

