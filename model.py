"""
Decoder-Only Transformer Implementation

A clean, modular implementation of a GPT-style decoder-only transformer
for character-level language modeling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for the Decoder Transformer model.

    Attributes:
        vocab_size: Size of the vocabulary.
        n_embed: Embedding dimension size.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        block_size: Maximum sequence/context length.
        dropout: Dropout probability for regularization.
        bias: Whether to use bias in linear layers.
    """
    vocab_size: int
    n_embed: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_embed % self.n_head != 0:
            raise ValueError(
                f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class TrainingConfig:
    """Configuration for training the transformer.

    Attributes:
        batch_size: Number of sequences per batch.
        max_iters: Maximum training iterations.
        eval_interval: How often to evaluate on validation set.
        eval_iters: Number of batches to average for evaluation.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW optimizer.
        grad_clip: Maximum gradient norm for clipping (None to disable).
        use_lr_scheduler: Whether to use cosine annealing LR scheduler.
        checkpoint_dir: Directory to save model checkpoints.
    """
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: Optional[float] = 1.0
    use_lr_scheduler: bool = True
    checkpoint_dir: Optional[Path] = None


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention module with causal masking.

    Implements scaled dot-product attention with multiple heads,
    allowing the model to jointly attend to information from
    different representation subspaces.

    Args:
        config: TransformerConfig containing model hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_size = config.n_embed // config.n_head

        # Combined Q, K, V projection for efficiency
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask to prevent attending to future tokens
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed).
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Compute Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention with causal masking
        scale = 1.0 / (self.head_size ** 0.5)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, n_head, T, head_size)

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with residual dropout
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    A two-layer MLP with an expansion factor of 4x,
    applied independently to each position.

    Args:
        config: TransformerConfig containing model hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias),
            nn.GELU(),  # GELU is more commonly used in modern transformers
            nn.Dropout(config.dropout),  # Dropout after activation
            nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed).
        """
        return self.net(x)


class Block(nn.Module):
    """Transformer block combining attention and feed-forward layers.

    Uses pre-LayerNorm architecture (norm before sublayer) with
    residual connections around each sublayer.

    Args:
        config: TransformerConfig containing model hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    """Decoder-only Transformer for language modeling.

    A GPT-style autoregressive transformer that predicts the next token
    given a sequence of previous tokens.

    Args:
        config: TransformerConfig containing model hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight tying between token embeddings and output projection
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized model with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using standard transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the transformer.

        Args:
            idx: Input token indices of shape (batch_size, seq_len).
            targets: Optional target token indices for computing loss.

        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided.

        Raises:
            ValueError: If sequence length exceeds block_size.
        """
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds maximum block_size {self.config.block_size}"
            )

        # Get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embed)
        pos_emb = self.position_embedding(
            torch.arange(T, device=self.device)
        )  # (T, n_embed)
        x = self.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            idx: Starting token indices of shape (batch_size, seq_len).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from top k most likely tokens.

        Returns:
            Token indices including generated tokens of shape
            (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens if needed
            idx_cond = idx[:, -self.config.block_size:]

            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Trainer:
    """Trainer for the DecoderTransformer model.

    Handles the training loop with support for:
    - Gradient clipping
    - Learning rate scheduling
    - Model checkpointing
    - Validation evaluation

    Args:
        model: The DecoderTransformer model to train.
        config: TrainingConfig with training hyperparameters.
        get_batch: Function that returns (x, y) batches for a given split.
        device: Device to train on.
    """

    def __init__(
        self,
        model: DecoderTransformer,
        config: TrainingConfig,
        get_batch: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device
    ) -> None:
        self.model = model
        self.config = config
        self.get_batch = get_batch
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_iters,
                eta_min=config.learning_rate / 10
            )

        # Tracking
        self.best_val_loss = float('inf')

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets.

        Returns:
            Dictionary with 'train' and 'val' average losses.
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            path: Directory to save checkpoint.
            is_best: Whether this is the best model so far.
        """
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'best_val_loss': self.best_val_loss,
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path / 'checkpoint.pt')
        if is_best:
            torch.save(checkpoint, path / 'best_model.pt')
            logger.info(f"Saved best model with val_loss={self.best_val_loss:.4f}")

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()

        for iter_num in range(self.config.max_iters):
            # Evaluate periodically
            if iter_num % self.config.eval_interval == 0 or iter_num == self.config.max_iters - 1:
                losses = self.estimate_loss()
                logger.info(
                    f"step {iter_num}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                # Checkpointing
                if self.config.checkpoint_dir:
                    is_best = losses['val'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = losses['val']
                    self.save_checkpoint(self.config.checkpoint_dir, is_best)

            # Training step
            xb, yb = self.get_batch('train')
            _, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()


def create_char_tokenizer(text: str) -> Tuple[Callable[[str], list], Callable[[list], str], int]:
    """Create character-level encoder and decoder functions.

    Args:
        text: Text to build vocabulary from.

    Returns:
        Tuple of (encode_fn, decode_fn, vocab_size).
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list:
        return [stoi[c] for c in s]

    def decode(tokens: list) -> str:
        return ''.join([itos[i] for i in tokens])

    return encode, decode, vocab_size


if __name__ == '__main__':
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    input_path = Path('input.txt')
    if not input_path.exists():
        raise FileNotFoundError(
            f"Training data file '{input_path}' not found. "
            "Please provide an input.txt file with training text."
        )

    text = input_path.read_text(encoding='utf-8')
    logger.info(f"Loaded {len(text)} characters from {input_path}")

    # Create tokenizer
    encode, decode, vocab_size = create_char_tokenizer(text)

    # Prepare data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model configuration
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        n_embed=384,
        n_head=6,
        n_layer=6,
        block_size=256,
        dropout=0.2,
    )

    # Training configuration
    train_config = TrainingConfig(
        batch_size=64,
        max_iters=5000,
        eval_interval=500,
        eval_iters=200,
        learning_rate=3e-4,
        grad_clip=1.0,
        use_lr_scheduler=True,
        checkpoint_dir=Path('checkpoints'),
    )

    # Batch function
    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - model_config.block_size, (train_config.batch_size,))
        x = torch.stack([data_split[i:i + model_config.block_size] for i in ix])
        y = torch.stack([data_split[i + 1:i + model_config.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Initialize model and trainer
    model = DecoderTransformer(model_config).to(device)
    trainer = Trainer(model, train_config, get_batch, device)

    # Train
    trainer.train()

    # Generate sample text
    logger.info("Generating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print("\n--- Generated Text ---")
    print(decode(generated[0].tolist()))
