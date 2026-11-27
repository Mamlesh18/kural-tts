import pytest

torch = pytest.importorskip("torch")

from training.models.vits.attentions import Encoder, MultiHeadAttention  # noqa: E402


def test_rel_to_abs_matches_bruteforce():
    length = 5
    x = torch.randn(1, 1, length, 2 * length - 1)
    out = MultiHeadAttention._rel_to_abs(x)
    for i in range(length):
        for j in range(length):
            assert out[0, 0, i, j] == x[0, 0, i, (j - i) + length - 1]


def test_abs_to_rel_matches_bruteforce():
    length = 4
    x = torch.randn(1, 1, length, length)
    out = MultiHeadAttention._abs_to_rel(x)
    assert out.shape == (1, 1, length, 2 * length - 1)
    for i in range(length):
        for j in range(length):
            assert out[0, 0, i, (j - i) + length - 1] == x[0, 0, i, j]


@pytest.mark.parametrize("length", [1, 3, 5, 12])
def test_relative_embeddings_window(length):
    mha = MultiHeadAttention(8, 8, n_heads=2, window_size=2)
    rel = mha._relative_embeddings(mha.emb_rel_k, length)
    assert rel.shape == (1, 2 * length - 1, 4)
    centre = length - 1
    for offset in range(-(length - 1), length):
        row = rel[0, centre + offset]
        if abs(offset) <= 2:
            assert torch.equal(row, mha.emb_rel_k[0, 2 + offset])
        else:
            assert torch.count_nonzero(row) == 0  # beyond the window: no embedding


def test_encoder_respects_mask():
    enc = Encoder(16, 32, n_heads=2, n_layers=2, kernel_size=3).eval()
    x = torch.randn(2, 16, 7)
    mask = torch.ones(2, 1, 7)
    mask[1, :, 4:] = 0
    out = enc(x, mask)
    assert out.shape == x.shape
    assert torch.all(out[1, :, 4:] == 0)
    # Padding must not change the valid part of the output.
    out_short = enc(x[1:, :, :4], torch.ones(1, 1, 4))
    assert torch.allclose(out[1:, :, :4], out_short, atol=1e-5)
