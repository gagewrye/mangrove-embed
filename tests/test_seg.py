import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mangrove_embed.seg import JaccardLoss, SegFormer  # noqa: E402

W = "nvidia/segformer-b0-finetuned-ade-512-512"


@pytest.fixture(scope="module")
def emb():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 64, 64)
    return x / x.norm(dim=1, keepdim=True)  # unit-length, like AlphaEarth


def _stem(m):
    return next(x for x in m.backbone.modules() if isinstance(x, torch.nn.Conv2d))


def test_native_64ch_keeps_pretrained_and_trains(emb):
    ref = SegFormer(num_classes=11, input_image_size=64, weights=W)
    m = SegFormer(num_classes=11, input_image_size=64, weights=W, in_channels=64)
    assert _stem(m).in_channels == 64
    a, b = ref.backbone.state_dict(), m.backbone.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if a[k].shape == b[k].shape)
    out = m(emb)
    assert out.shape == (2, 11, 64, 64)
    loss = JaccardLoss(num_classes=11)(out, torch.randint(0, 11, (2, 64, 64)))
    loss.backward()
    assert _stem(m).weight.grad.abs().sum() > 0


def test_binary_mean_init_and_stride(emb):
    m = SegFormer(num_classes=1, input_image_size=64, weights=W, in_channels=64,
                  stem_init="mean", stem_stride=2)
    assert _stem(m).stride == (2, 2)
    out = m(emb)
    assert out.shape == (2, 1, 64, 64)
    assert torch.isfinite(JaccardLoss(num_classes=1)(out, torch.randint(0, 2, (2, 64, 64))))


def test_channel_mismatch_raises():
    m = SegFormer(num_classes=1, input_image_size=64, weights=W, in_channels=64)
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 64, 64))
