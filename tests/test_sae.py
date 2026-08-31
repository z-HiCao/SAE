import torch

from sae_repro.sae.models import BatchTopKSAE, MatryoshkaBatchTopKSAE, ReLUSAE, UniversalSAE


def test_relu_sae_shapes() -> None:
    """基础 SAE 的 latent 和重建形状必须正确。"""
    model = ReLUSAE(input_dim=8, latent_dim=24, l1_coefficient=0.01)
    values = torch.randn(6, 8)
    latent, reconstruction = model(values)
    assert latent.shape == (6, 24)
    assert reconstruction.shape == values.shape
    assert torch.all(latent >= 0)


def test_batch_topk_budget() -> None:
    """训练态非零激活不能超过 B×K。"""
    model = BatchTopKSAE(input_dim=8, latent_dim=16, k=3)
    model.train()
    latent, _ = model(torch.randn(5, 8))
    assert int((latent > 0).sum()) <= 5 * 3


def test_matryoshka_loss_is_scalar() -> None:
    """Matryoshka 多前缀损失应汇总为标量。"""
    model = MatryoshkaBatchTopKSAE(8, 16, 3, [0.25, 0.5, 1.0])
    loss, parts = model.compute_loss(torch.randn(5, 8))
    assert loss.ndim == 0
    assert "matryoshka_loss" in parts


def test_universal_shapes() -> None:
    """两个模型应编码到同一 latent 宽度并各自重建原维度。"""
    model = UniversalSAE([8, 11], latent_dim=20, k=4)
    left = torch.randn(7, 8)
    right = torch.randn(7, 11)
    code = model.encode(0, left)
    assert code.shape == (7, 20)
    assert model.decode(0, code).shape == left.shape
    assert model.decode(1, code).shape == right.shape

