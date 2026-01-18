"""
Unit tests for the Decoder Transformer model.

Run with: pytest test_model.py -v
"""

import pytest
import torch

from model import (
    TransformerConfig,
    TrainingConfig,
    MultiHeadAttention,
    FeedForward,
    Block,
    DecoderTransformer,
    create_char_tokenizer,
)


class TestTransformerConfig:
    """Tests for TransformerConfig validation."""

    def test_valid_config(self) -> None:
        """Test that valid config is accepted."""
        config = TransformerConfig(
            vocab_size=100,
            n_embed=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            dropout=0.1,
        )
        assert config.vocab_size == 100
        assert config.n_embed == 64

    def test_invalid_n_embed_n_head_ratio(self) -> None:
        """Test that n_embed must be divisible by n_head."""
        with pytest.raises(ValueError, match="must be divisible by n_head"):
            TransformerConfig(vocab_size=100, n_embed=63, n_head=4)

    def test_invalid_vocab_size(self) -> None:
        """Test that vocab_size must be positive."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            TransformerConfig(vocab_size=0)

    def test_invalid_block_size(self) -> None:
        """Test that block_size must be positive."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            TransformerConfig(vocab_size=100, block_size=0)

    def test_invalid_dropout(self) -> None:
        """Test that dropout must be in [0, 1)."""
        with pytest.raises(ValueError, match="dropout must be in"):
            TransformerConfig(vocab_size=100, dropout=1.0)
        with pytest.raises(ValueError, match="dropout must be in"):
            TransformerConfig(vocab_size=100, dropout=-0.1)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=100,
            n_embed=32,
            n_head=4,
            block_size=16,
            dropout=0.0,  # Disable dropout for deterministic tests
        )

    def test_output_shape(self, config: TransformerConfig) -> None:
        """Test that output shape matches input shape."""
        attn = MultiHeadAttention(config)
        x = torch.randn(2, 10, config.n_embed)
        y = attn(x)
        assert y.shape == x.shape

    def test_causal_mask_exists(self, config: TransformerConfig) -> None:
        """Test that causal mask is registered as buffer."""
        attn = MultiHeadAttention(config)
        assert hasattr(attn, 'causal_mask')
        assert attn.causal_mask.shape == (1, 1, config.block_size, config.block_size)

    def test_causal_mask_is_lower_triangular(self, config: TransformerConfig) -> None:
        """Test that causal mask is lower triangular."""
        attn = MultiHeadAttention(config)
        mask = attn.causal_mask.squeeze()
        assert torch.allclose(mask, torch.tril(torch.ones_like(mask)))


class TestFeedForward:
    """Tests for FeedForward module."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=100,
            n_embed=32,
            n_head=4,
            dropout=0.0,
        )

    def test_output_shape(self, config: TransformerConfig) -> None:
        """Test that output shape matches input shape."""
        ffwd = FeedForward(config)
        x = torch.randn(2, 10, config.n_embed)
        y = ffwd(x)
        assert y.shape == x.shape


class TestBlock:
    """Tests for transformer Block."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=100,
            n_embed=32,
            n_head=4,
            block_size=16,
            dropout=0.0,
        )

    def test_output_shape(self, config: TransformerConfig) -> None:
        """Test that output shape matches input shape."""
        block = Block(config)
        x = torch.randn(2, 10, config.n_embed)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_connection(self, config: TransformerConfig) -> None:
        """Test that block uses residual connections (output != zero)."""
        block = Block(config)
        block.eval()
        x = torch.randn(2, 10, config.n_embed)
        y = block(x)
        # Output should be different from input but not zero
        assert not torch.allclose(y, x)
        assert not torch.allclose(y, torch.zeros_like(y))


class TestDecoderTransformer:
    """Tests for DecoderTransformer model."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=100,
            n_embed=32,
            n_head=4,
            n_layer=2,
            block_size=16,
            dropout=0.0,
        )

    @pytest.fixture
    def model(self, config: TransformerConfig) -> DecoderTransformer:
        return DecoderTransformer(config)

    def test_forward_without_targets(self, model: DecoderTransformer) -> None:
        """Test forward pass without targets returns logits and None loss."""
        idx = torch.randint(0, 100, (2, 10))
        logits, loss = model(idx)
        assert logits.shape == (2, 10, 100)
        assert loss is None

    def test_forward_with_targets(self, model: DecoderTransformer) -> None:
        """Test forward pass with targets returns logits and loss."""
        idx = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 10, 100)
        assert loss is not None
        assert loss.ndim == 0  # Scalar loss

    def test_sequence_length_validation(self, model: DecoderTransformer) -> None:
        """Test that sequence length exceeding block_size raises error."""
        idx = torch.randint(0, 100, (1, 20))  # block_size is 16
        with pytest.raises(ValueError, match="exceeds maximum block_size"):
            model(idx)

    def test_generate_output_length(self, model: DecoderTransformer) -> None:
        """Test that generate produces correct number of new tokens."""
        model.eval()
        idx = torch.randint(0, 100, (1, 5))
        max_new_tokens = 10
        output = model.generate(idx, max_new_tokens=max_new_tokens)
        assert output.shape == (1, 5 + max_new_tokens)

    def test_generate_with_temperature(self, model: DecoderTransformer) -> None:
        """Test generation with different temperatures."""
        model.eval()
        idx = torch.randint(0, 100, (1, 5))

        # Low temperature should work
        output_low = model.generate(idx.clone(), max_new_tokens=5, temperature=0.1)
        assert output_low.shape == (1, 10)

        # High temperature should work
        output_high = model.generate(idx.clone(), max_new_tokens=5, temperature=2.0)
        assert output_high.shape == (1, 10)

    def test_generate_with_top_k(self, model: DecoderTransformer) -> None:
        """Test generation with top-k sampling."""
        model.eval()
        idx = torch.randint(0, 100, (1, 5))
        output = model.generate(idx, max_new_tokens=5, top_k=10)
        assert output.shape == (1, 10)

    def test_device_property(self, model: DecoderTransformer) -> None:
        """Test that device property returns correct device."""
        assert model.device == torch.device('cpu')

    def test_weight_tying(self, model: DecoderTransformer) -> None:
        """Test that token embedding and lm_head share weights."""
        assert model.token_embedding.weight is model.lm_head.weight


class TestCreateCharTokenizer:
    """Tests for character tokenizer creation."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode/decode is reversible."""
        text = "hello world"
        encode, decode, vocab_size = create_char_tokenizer(text)

        encoded = encode(text)
        decoded = decode(encoded)
        assert decoded == text

    def test_vocab_size(self) -> None:
        """Test that vocab_size matches unique characters."""
        text = "aabbcc"
        encode, decode, vocab_size = create_char_tokenizer(text)
        assert vocab_size == 3  # a, b, c

    def test_sorted_vocabulary(self) -> None:
        """Test that vocabulary is sorted."""
        text = "cba"
        encode, decode, vocab_size = create_char_tokenizer(text)
        # 'a' should be index 0, 'b' index 1, 'c' index 2
        assert encode("abc") == [0, 1, 2]


class TestModelTraining:
    """Integration tests for model training."""

    @pytest.fixture
    def small_config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=10,
            n_embed=16,
            n_head=2,
            n_layer=1,
            block_size=8,
            dropout=0.0,
        )

    def test_loss_decreases(self, small_config: TransformerConfig) -> None:
        """Test that loss decreases during training."""
        model = DecoderTransformer(small_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # Generate random data
        data = torch.randint(0, 10, (100,))

        initial_loss = None
        final_loss = None

        for i in range(50):
            ix = torch.randint(0, len(data) - small_config.block_size, (4,))
            x = torch.stack([data[j:j + small_config.block_size] for j in ix])
            y = torch.stack([data[j + 1:j + small_config.block_size + 1] for j in ix])

            _, loss = model(x, y)

            if i == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        assert final_loss < initial_loss, "Loss should decrease during training"

    def test_gradient_flow(self, small_config: TransformerConfig) -> None:
        """Test that gradients flow through all parameters."""
        model = DecoderTransformer(small_config)

        x = torch.randint(0, 10, (2, 6))
        y = torch.randint(0, 10, (2, 6))

        _, loss = model(x, y)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
