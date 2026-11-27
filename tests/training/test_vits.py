import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.models.vits.commons import sequence_mask  # noqa: E402
from training.models.vits.discriminator import MultiPeriodDiscriminator  # noqa: E402
from training.models.vits.losses import (  # noqa: E402
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from training.models.vits.mas import maximum_path  # noqa: E402
from training.models.vits.synthesizer import ModelConfig, SynthesizerTrn  # noqa: E402


def tiny_config(**kw) -> ModelConfig:
    base = dict(
        n_vocab=75,
        n_langs=2,
        spec_channels=513,
        segment_frames=8,
        inter_channels=16,
        hidden_channels=16,
        filter_channels=32,
        n_heads=2,
        n_layers=2,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
        upsample_initial_channel=32,
        posterior_layers=2,
        flow_layers=2,
        n_flows=2,
        dp_filter_channels=16,
    )
    base.update(kw)
    return ModelConfig(**base)


def _bruteforce_path(value: np.ndarray, t_y: int, t_x: int) -> np.ndarray:
    """Reference O(T_y*T_x) DP from the original VITS Cython code."""
    q = np.full((t_y, t_x), -1e9)
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = -1e9 if x == y else (q[y - 1, x] if y > 0 else -1e9)
            if x == 0:
                v_prev = 0.0 if y == 0 else -1e9
            else:
                v_prev = q[y - 1, x - 1] if y > 0 else -1e9
            q[y, x] = max(v_cur, v_prev) + value[y, x]
    path = np.zeros((t_y, t_x))
    x = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, x] = 1
        if x != 0 and (x == y or q[y - 1, x] < q[y - 1, x - 1]):
            x -= 1
    return path


def test_mas_is_monotonic_and_matches_reference():
    rng = np.random.default_rng(0)
    t_ys, t_xs = [12, 9], [5, 3]
    value = rng.normal(size=(2, 12, 5)).astype(np.float32)
    mask = np.zeros_like(value)
    for i, (ty, tx) in enumerate(zip(t_ys, t_xs, strict=True)):
        mask[i, :ty, :tx] = 1
    path = maximum_path(torch.from_numpy(value), torch.from_numpy(mask)).numpy()
    for i, (ty, tx) in enumerate(zip(t_ys, t_xs, strict=True)):
        p = path[i, :ty, :tx]
        assert (p.sum(1) == 1).all(), "each frame maps to exactly one token"
        assert (p.sum(0) >= 1).all(), "every token gets at least one frame"
        idx = p.argmax(1)
        assert (np.diff(idx) >= 0).all() and (np.diff(idx) <= 1).all()
        assert path[i].sum() == ty
        np.testing.assert_array_equal(p, _bruteforce_path(value[i], ty, tx))


def test_training_forward_shapes_and_losses():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = SynthesizerTrn(cfg)
    disc = MultiPeriodDiscriminator()
    x = torch.randint(4, 75, (2, 10))
    lang = torch.randint(0, 2, (2, 10))
    x_len = torch.tensor([10, 7])
    y = torch.rand(2, 513, 40)
    y_len = torch.tensor([40, 30])

    o, l_length, attn, _ids, _x_mask, y_mask, (_z, z_p, m_p, logs_p, _m_q, logs_q) = model(
        x, x_len, lang, y, y_len
    )
    hop = model.hop_length
    assert o.shape == (2, 1, cfg.segment_frames * hop)
    assert attn.shape == (2, 1, 40, 10)
    assert torch.allclose(attn.sum((2, 3)).squeeze(), y_len.float())

    wav = torch.randn(2, 1, cfg.segment_frames * hop).clamp(-1, 1)
    y_d_r, y_d_g, fmap_r, fmap_g = disc(wav, o.detach())
    loss_d = discriminator_loss(y_d_r, y_d_g)
    _, y_d_g, fmap_r, fmap_g = disc(wav, o)
    loss_g = (
        generator_loss(y_d_g)
        + feature_loss(fmap_r, fmap_g)
        + l_length.sum()
        + kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
    )
    assert torch.isfinite(loss_d) and torch.isfinite(loss_g)
    loss_g.backward()
    assert model.enc_p.lang_emb.weight.grad is not None


def test_inference_length_follows_durations():
    torch.manual_seed(0)
    model = SynthesizerTrn(tiny_config()).eval()
    x = torch.randint(4, 75, (1, 6))
    lang = torch.zeros_like(x)
    with torch.no_grad():
        o1, _attn, y_mask = model.infer(x, torch.tensor([6]), lang, length_scale=1.0)
        o2, _, _ = model.infer(x, torch.tensor([6]), lang, length_scale=2.0)
    frames = int(y_mask.sum())
    assert o1.shape[-1] == frames * model.hop_length
    assert o2.shape[-1] > o1.shape[-1]


def test_remove_weight_norm_preserves_output():
    torch.manual_seed(0)
    model = SynthesizerTrn(tiny_config()).eval()
    x = torch.randint(4, 75, (1, 5))
    lang = torch.ones_like(x)
    with torch.no_grad():
        torch.manual_seed(1)
        before, _, _ = model.infer(x, torch.tensor([5]), lang, noise_scale=0.0)
        model.remove_weight_norm()
        torch.manual_seed(1)
        after, _, _ = model.infer(x, torch.tensor([5]), lang, noise_scale=0.0)
    assert torch.allclose(before, after, atol=1e-5)


def test_sequence_mask():
    m = sequence_mask(torch.tensor([1, 3]))
    assert m.tolist() == [[True, False, False], [True, True, True]]
