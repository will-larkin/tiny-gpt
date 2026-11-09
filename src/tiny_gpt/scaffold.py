"""Tiny GPT in PyTorch — end-to-end demo scaffold."""

import torch
from tiny_gpt.model import CharTokenizer, GPTConfig, TinyGPT, make_batches


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = (
        "hello world. this is a tiny gpt demo built from pytorch.\n"
        "the quick brown fox jumps over the lazy dog.\n"
        "transformers learn to predict the next character.\n"
    ) * 8

    tokenizer = CharTokenizer(corpus)
    data = tokenizer.encode(corpus)
    print(f"vocab_size={tokenizer.vocab_size}, corpus_tokens={len(data)}")
    print(f"sample decode: {tokenizer.decode(data[:30])!r}")

    block_size = 16
    batch_size = 8
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2
    num_steps = 800
    learning_rate = 2e-3

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )
    model = TinyGPT(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"number of model parameters: {num_params}")

    # Inspect initial loss
    inputs, targets = make_batches(data, batch_size, block_size, device=device)
    print(f"batch shapes: inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}")
    logits, init_loss = model(inputs, targets)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"initial loss: {init_loss.item():.4f}")

    # Training with AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    model.train()
    for step in range(num_steps):
        xb, yb = make_batches(data, batch_size, block_size, device=device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"loss history (first/last): {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Text Generation
    prompt = "the "
    context_tokens = tokenizer.encode(prompt, device=device).unsqueeze(0)
    generated_tokens = model.generate(
        idx=context_tokens,
        max_new_tokens=40,
        temperature=0.8,
        top_k=5,
    )
    generated_text = tokenizer.decode(generated_tokens[0])
    print(f"prompt: {prompt!r}")
    print(f"generated: {generated_text!r}")


if __name__ == "__main__":
    main()
