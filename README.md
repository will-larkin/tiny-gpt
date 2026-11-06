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

---

Built on Deep-ML.
