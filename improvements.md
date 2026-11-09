# Tiny GPT: Modernization & Extension Roadmap

A high-level, ordered sequence of potential architectural improvements, tooling upgrades, and expansion options for evolving `tiny-gpt` into a modern PyTorch-based language model.

---

## 1. Environment & Runtime Foundation
*Establish the runtime platform, package manager, and PyTorch ecosystem support.*

- **Option 1A: Python 3.12 Target with `uv` (Selected & Implemented)**
  - *Pros:* Fully compatible with official stable PyTorch wheels (`torch>=2.2.0`), deterministic resolution via `uv.lock`, fast virtual environment creation.
  - *Cons:* Requires pinning to `<3.13`.
- **Option 1B: System Python (3.11) with loose requirement (`>=3.11`)**
  - *Pros:* Adapts directly to whatever Python version is on the host.
  - *Cons:* Less reproducible across developer environments.

---

## 2. Transformer Backbone & Architecture
*Upgrade from the legacy 2018 GPT-2 design to modern 2024+ open-weights LLM architectures (e.g. LLaMA 3, Gemma, Mistral).*

- **Option 2A: Full Modern LLM Stack by Default (Selected)**
  - **RMSNorm** instead of LayerNorm (faster, eliminates mean computation).
  - **RoPE (Rotary Position Embeddings)** replacing fixed/learned positional tables (enables sequence length extrapolation and relative position awareness).
  - **SwiGLU Activation** replacing standard GELU MLP (demonstrated empirical perplexity gains across all modern LLMs).
  - **PyTorch `F.scaled_dot_product_attention` (SDPA)** for FlashAttention hardware acceleration.
- **Option 2B: Strict Classic GPT-2 Architecture**
  - Standard LayerNorm, learned positional embeddings, GELU activations, and standard multi-head attention.
  - *Pros:* Faithful reproduction of original GPT-2 papers and minimal conceptual complexity.
  - *Cons:* Slower, less performant, and does not reflect current state-of-the-art transformer design.
- **Option 2C: Configurable Hybrid (`GPTConfig` toggles)**
  - Parameterized flags allowing runtime switching between LayerNorm vs RMSNorm, Learned vs RoPE, and GELU vs SwiGLU.
  - *Pros:* Excellent for comparative benchmarking and ablations.
  - *Cons:* Additional branching and abstraction overhead in the model code.

---

## 3. Tokenization Strategy
*Convert raw text into model-consumable token sequences.*

- **Option 3A: Dual-Engine Tokenizer (Selected)**
  - Includes a zero-dependency **Character Tokenizer** (for instant toy experiments without external downloads) and OpenAI's **`tiktoken` BPE** (for standard subword vocabularies such as `gpt2` or `cl100k_base`).
  - *Pros:* Best of both worlds—zero barrier for quick testing, production-grade compression for real datasets.
- **Option 3B: BPE-Only via `tiktoken`**
  - *Pros:* Consistent vocabulary distribution and faster training throughput due to ~4x text compression.
  - *Cons:* External tokenization dependency required even for minimal single-sentence checks.
- **Option 3C: Self-Trained Tokenizer (minBPE or HuggingFace `tokenizers`)**
  - Implement or bundle a training script to generate custom BPE vocabularies tailored to specific domain datasets.
  - *Pros:* Maximum educational depth on how byte-pair encoding algorithms work from scratch.
  - *Cons:* Extra preprocessing pipeline complexity before training can begin.

---

## 4. Inference Engine & Text Generation
*Generate tokens autoregressively from prompts.*

- **Option 4A: Fast KV-Cache Decoding + Streaming Generator (Selected)**
  - Retains key/value tensors from past positions so each new token requires only an $O(1)$ attention step rather than recomputing the full prefix ($O(T^2)$).
  - Includes temperature scaling, Top-k, and **Top-p (Nucleus) sampling**, with a Python generator interface for streaming output token-by-token.
- **Option 4B: Stateless Autoregressive Generator**
  - Re-evaluates the entire context window on every step without caching.
  - *Pros:* Very simple to implement and debug.
  - *Cons:* Decimates generation speed as sequence length increases.
- **Option 4C: Interactive REPL / Chat Interface**
  - Built on top of the KV-cache generator to allow real-time conversational prompting directly in the terminal.

---

## 5. Training Pipeline & Optimization
*Manage forward/backward passes, optimization dynamics, and persistent checkpoints.*

- **Option 5A: End-to-End Pretraining Pipeline (Selected)**
  - PyTorch `Dataset` and `DataLoader` for streaming/chunking text.
  - **AdamW optimizer** with weight decay separation (excluding norms and biases).
  - **Cosine learning rate schedule** with linear warmup.
  - **Automatic Mixed Precision (AMP)** (`torch.autocast`) for faster execution on modern hardware.
  - **Gradient clipping** to prevent loss explosions.
  - **Model checkpointing** (saving and resuming `.pt` weights, optimizer states, and training step counts).
- **Option 5B: Minimal Training Loop**
  - Standard PyTorch loop on a single text file without learning rate scheduling, mixed precision, or checkpointing.
  - *Pros:* Minimal code footprint.
  - *Cons:* Not practical for training beyond small toy runs.
- **Option 5C: Multi-Stage Pretraining + Instruction Fine-Tuning (SFT)**
  - Adds a prompt-response formatting pipeline (e.g. ChatML format) to fine-tune pretrained models on instruction datasets (Alpaca / JSONL).

---

## 6. Codebase Architecture & CLI
*Structure files and developer interfaces for clarity, modularity, and maintainability.*

- **Option 6A: Modular Package with Unified CLI (Selected)**
  - Packaged under `src/tiny_gpt/` (`model.py`, `config.py`, `tokenizer.py`, `dataset.py`, `trainer.py`, `generate.py`, `cli.py`).
  - Single executable entry point (`tiny-gpt train`, `tiny-gpt sample`, `tiny-gpt demo`).
  - *Pros:* Clean separation of concerns, reusable library import, and intuitive CLI.
- **Option 6B: Flat Script Collection**
  - Top-level standalone scripts (`train.py`, `sample.py`, `model.py`).
  - *Pros:* Direct file execution without package installation.
  - *Cons:* Code duplication across scripts and weaker import boundaries.

---

## 7. Verification & Automated Testing
*Validate model correctness, numerical equivalence, and training stability.*

- **Option 7A: Comprehensive Pytest Suite (Selected)**
  - **Shape validation:** Input-to-output dimension verification through all submodules.
  - **RoPE correctness:** Ensures rotation properties hold across sequence positions.
  - **KV-Cache equivalence:** Asserts that KV-cached step-by-step decoding produces identical logits to full-sequence forward passes.
  - **Single-batch overfit test:** Confirms loss decreases to near-zero on a fixed batch, verifying backprop and optimizer updates.
- **Option 7B: Integration Demo Script Only**
  - A single script that runs a small end-to-end forward/backward step.
  - *Pros:* Quick to write.
  - *Cons:* Doesn't catch edge-case regressions in caching or rotary math.

---

## 8. Advanced Scaling & Deployment Frontiers (Future Enhancements)
*Extensions to explore once the core PyTorch model is in place.*

- **8.1 Grouped-Query Attention (GQA / MQA)**: Reduce KV-cache memory footprint by sharing key/value heads across query head groups (as in LLaMA 3 / Mistral).
- **8.2 Context Window Expansion**: Implement RoPE frequency scaling (Linear interpolation or YaRN) to scale sequence context beyond training limits.
- **8.3 Optimization & Compilation**:
  - `torch.compile` graph optimization for kernel fusion.
  - FP8 / INT8 / INT4 weight-only quantization for lightweight deployment.
  - Export to ONNX or GGUF formats.
- **8.4 Distributed Pretraining**: Multi-GPU scaling using PyTorch Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).
