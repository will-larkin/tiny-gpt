# Tiny GPT From Scratch

Build a minimal GPT-style language model end-to-end in pure NumPy, from character tokenization and softmax up through multi-head self-attention, transformer blocks, Adam, and sampling. Every forward and backward pass is derived and implemented by hand so you understand exactly how a transformer trains and generates text.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_char_vocab
- [x] **2.** build_id_to_char
- [x] **3.** encode_text
- [x] **4.** decode_ids
- [x] **5.** make_batches
- [x] **6.** token_embedding_lookup
- [x] **7.** add_positional_embeddings
- [x] **8.** linear_projection
- [x] **9.** compute_attention_scores
- [x] **10.** scale_attention_scores
- [x] **11.** apply_causal_mask
- [x] **12.** softmax_attention_weights
- [x] **13.** attention_context
- [x] **14.** split_heads
- [x] **15.** merge_heads
- [x] **16.** project_qkv
- [x] **17.** multi_head_scaled_dot_product_attention

---

Built on Deep-ML.
