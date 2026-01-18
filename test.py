"""
Quick training simulation for the Decoder Transformer.

This script demonstrates the model training on a tiny dataset
to verify everything works correctly.
"""

import torch

from model import (
    DecoderTransformer,
    TransformerConfig,
    create_char_tokenizer,
)


def main() -> None:
    """Run a quick training simulation."""
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration for a fast, toy example
    batch_size = 8
    max_iters = 20
    learning_rate = 1e-3

    # Toy dataset
    text = "hello world, this is a test to see how the model trains."
    encode, decode, vocab_size = create_char_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)

    # Model configuration with small values for quick testing
    config = TransformerConfig(
        vocab_size=vocab_size,
        n_embed=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        dropout=0.1,
    )

    # Data loader
    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data) - config.block_size, (batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Initialize model
    model = DecoderTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("--- Starting a quick training simulation ---")
    initial_loss = None
    final_loss = None

    for i in range(max_iters):
        xb, yb = get_batch()
        _, loss = model(xb, yb)

        if i == 0:
            initial_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"Iteration {i}/{max_iters}, Loss: {loss.item():.4f}")

        if i == max_iters - 1:
            final_loss = loss.item()

    print("\n--- Simulation finished ---")
    if initial_loss and final_loss and final_loss < initial_loss:
        print(f"Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
        print("Model is learning correctly!")
    else:
        print("Warning: Loss did not decrease as expected")

    # Generate some text
    print("\n--- Generating text from the test model ---")
    start_context = torch.tensor(encode("hello "), dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = model.generate(start_context, max_new_tokens=50, temperature=1.0)
    generated_text = decode(generated_ids[0].tolist())
    print(f"Generated text: {generated_text}")


if __name__ == '__main__':
    main()
