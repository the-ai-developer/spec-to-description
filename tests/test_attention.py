"""Multi-head attention, encoder/decoder blocks and the causal mask.

The point of these modules is that the maths can be checked by hand, so the
tests check properties rather than shapes alone: that a head really attends
where it should, that the mask blocks the future, and that the residual
connections leave the block well-behaved.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec2desc.attention import (  # noqa: E402
    DecoderBlock,
    EncoderBlock,
    FeedForward,
    MultiHeadAttention,
    causal_mask,
)


class TestCausalMask:
    def test_is_lower_triangular(self):
        m = causal_mask(4)
        assert m.shape == (4, 4)
        assert bool(m[0, 0])                       # position 0 sees itself
        assert not bool(m[0, 1])                   # but not the future
        for i in range(4):
            for j in range(i + 1):
                assert bool(m[i, j]), f"({i},{j}) should be attendable"
            for j in range(i + 1, 4):
                assert not bool(m[i, j]), f"({i},{j}) should be masked"

    def test_dtypes_are_boolean(self):
        assert causal_mask(3).dtype == torch.bool


class TestMultiHeadAttention:
    def test_shapes_round_trip(self):
        attn = MultiHeadAttention(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        assert attn(x).shape == (2, 5, 16)

    def test_d_model_must_divide_by_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=10, n_heads=4)

    def test_weights_sum_to_one(self):
        attn = MultiHeadAttention(d_model=8, n_heads=2)
        _, w = attn(torch.randn(1, 4, 8), return_weights=True)
        assert w.shape == (1, 2, 4, 4)              # [B, heads, Tq, Tk]
        assert torch.allclose(w.sum(-1), torch.ones(1, 2, 4), atol=1e-5)

    def test_causal_mask_blocks_the_future(self):
        """Changing a future token must not change earlier outputs."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(d_model=8, n_heads=2).eval()
        mask = causal_mask(5)
        x = torch.randn(1, 5, 8)
        y1 = attn(x, mask=mask)

        x2 = x.clone()
        x2[0, 4] = torch.randn(8) * 10            # scramble the last position
        y2 = attn(x2, mask=mask)

        # positions 0..3 cannot attend to position 4, so they must be identical
        assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-6)
        assert not torch.allclose(y1[:, 4], y2[:, 4], atol=1e-6)

    def test_cross_attention_shape(self):
        attn = MultiHeadAttention(d_model=8, n_heads=2)
        q = torch.randn(2, 3, 8)
        kv = torch.randn(2, 7, 8)                  # encoder states are longer
        assert attn(q, kv).shape == (2, 3, 8)

    def test_permutation_equivariance_without_mask(self):
        """Attention has no positional bias, so reordering keys reorders outputs."""
        torch.manual_seed(1)
        attn = MultiHeadAttention(d_model=8, n_heads=2).eval()
        x = torch.randn(1, 4, 8)
        perm = torch.tensor([2, 0, 3, 1])
        base = attn(x)
        permuted = attn(x[:, perm])
        assert torch.allclose(base[:, perm], permuted, atol=1e-5)


class TestBlocks:
    def test_encoder_block_shape(self):
        block = EncoderBlock(d_model=16, n_heads=4, d_ff=32)
        assert block(torch.randn(2, 5, 16)).shape == (2, 5, 16)

    def test_decoder_block_shape_and_weights(self):
        block = DecoderBlock(d_model=16, n_heads=4, d_ff=32)
        memory = torch.randn(2, 9, 16)
        tgt = torch.randn(2, 4, 16)
        out, w = block(tgt, memory, tgt_mask=causal_mask(4), return_weights=True)
        assert out.shape == (2, 4, 16)
        # cross-attention is over the encoder states
        assert w.shape == (2, 4, 4, 9)

    def test_residual_connections_keep_scale_bounded(self):
        """Post-norm blocks with a residual path should not blow up."""
        torch.manual_seed(2)
        block = EncoderBlock(d_model=16, n_heads=4, d_ff=32).eval()
        x = torch.randn(2, 32, 16) * 50
        assert torch.isfinite(block(x)).all()
        assert block(x).abs().max() < x.abs().max()

    def test_feed_forward_is_position_wise(self):
        ff = FeedForward(d_model=8, d_ff=16)
        x = torch.randn(2, 5, 8)
        out = ff(x)
        # changing position 0 must not affect position 1
        x2 = x.clone()
        x2[:, 0] = torch.randn(8)
        out2 = ff(x2)
        assert torch.allclose(out[:, 1], out2[:, 1], atol=1e-6)

    def test_gradients_flow_to_both_projections(self):
        block = DecoderBlock(d_model=8, n_heads=2, d_ff=16)
        memory = torch.randn(1, 5, 8, requires_grad=True)
        tgt = torch.randn(1, 3, 8)
        block(tgt, memory).sum().backward()
        assert memory.grad is not None
        assert torch.isfinite(memory.grad).all()
        assert memory.grad.abs().sum() > 0
