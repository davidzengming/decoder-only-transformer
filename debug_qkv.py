"""
Debug script for visualizing Q/K/V operations in the transformer.

This script provides detailed tensor shape tracking through the
attention mechanism to help understand how the transformer processes input.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import TransformerConfig, create_char_tokenizer


class DebugMultiHeadAttention(nn.Module):
    """Multi-Head Attention with debug print statements.

    This is a standalone version of MultiHeadAttention that prints
    tensor shapes at each step for educational purposes.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_size = config.n_embed // config.n_head

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        print(f"    [ATTN] Input to Attention Head:      shape={x.shape}")

        # Q, K, V Calculation
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        print(f"    [ATTN] After splitting into Q, K, V:  q={q.shape}, k={k.shape}, v={v.shape}")

        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        print(f"    [ATTN] After reshaping for heads:    q={q.shape} (Batch, n_heads, Time, head_size)")

        # Attention Score Calculation
        scale = 1.0 / (self.head_size ** 0.5)
        att = (q @ k.transpose(-2, -1)) * scale
        print(f"    [ATTN] Attention scores (affinities): shape={att.shape} (How much each token attends to others)")

        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        print(f"    [ATTN] After applying causal mask (hiding future tokens)")

        att = F.softmax(att, dim=-1)
        print(f"    [ATTN] After softmax (scores become probabilities)")

        y = att @ v
        print(f"    [ATTN] Output after multiplying by V: shape={y.shape} (Weighted sum of values)")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        print(f"    [ATTN] Final output of Attention Head:  shape={y.shape}")
        return y


class DebugFeedForward(nn.Module):
    """Feed-forward network (no debug output needed here)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DebugBlock(nn.Module):
    """Transformer block with debug output."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn = DebugMultiHeadAttention(config)
        self.ffwd = DebugFeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\n  [BLOCK] Entering Transformer Block")
        print(f"  [BLOCK] Input shape: {x.shape}")
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        print(f"  [BLOCK] Output shape: {x.shape}")
        return x


def main() -> None:
    """Walk through the transformer with debug output."""
    # Configuration for debug run
    config = TransformerConfig(
        vocab_size=29,  # Will be updated based on actual vocab
        n_embed=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        dropout=0.1,
    )

    device = torch.device('cpu')  # Force CPU for simplicity in debugging

    # Input Text and Vocabulary
    text = "hello"
    vocab_text = "abcdefghijklmnopqrstuvwxyz ,."  # A bigger vocab
    encode, decode, vocab_size = create_char_tokenizer(vocab_text)

    # Update config with actual vocab size
    config = TransformerConfig(
        vocab_size=vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
    )

    input_ids = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)

    print("--- STEP 1: INPUT & EMBEDDINGS ---")
    print(f"Original text: '{text}'")
    print(f"Input tensor (token IDs): {input_ids}, shape={input_ids.shape}")

    # Model Components
    token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
    position_embedding = nn.Embedding(config.block_size, config.n_embed)

    tok_emb = token_embedding(input_ids)
    pos_emb = position_embedding(torch.arange(input_ids.shape[1], device=device))
    x = tok_emb + pos_emb

    print(f"Token embeddings shape:   {tok_emb.shape}")
    print(f"Position embeddings shape: {pos_emb.shape}")
    print(f"Final input to blocks (tok_emb + pos_emb): shape={x.shape}")

    # Transformer Blocks
    blocks = nn.Sequential(*[DebugBlock(config) for _ in range(config.n_layer)])
    y = blocks(x)

    # Final Output
    ln_f = nn.LayerNorm(config.n_embed)
    lm_head = nn.Linear(config.n_embed, config.vocab_size)
    y = lm_head(ln_f(y))

    print("\n--- STEP 2: FINAL OUTPUT ---")
    print(f"Output from blocks before final layer: shape={x.shape}")
    print(f"Final output (logits) shape: {y.shape} (Batch, Time, Vocab_Size)")
    print(f"\nEach of the {input_ids.shape[1]} tokens in the sequence now has a vector of {config.vocab_size} logits,")
    print("predicting the probability of the next token in the vocabulary.")


if __name__ == '__main__':
    main()
