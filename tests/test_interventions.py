import math

import torch

from sae_repro.metrics.interventions import positive_quantile
from sae_repro.models.clip_zero_shot import clip_zero_shot_logits


def test_positive_quantile_ignores_zeros() -> None:
    """稀疏 latent 的干预值必须由正激活而不是大量零值决定。"""
    values = torch.tensor([0.0, 0.0, 0.0, 2.0, 4.0])
    assert math.isclose(positive_quantile(values, 0.5), 3.0)


def test_clip_logits_restore_standardized_embedding() -> None:
    """标准化空间还原后应按 CLIP 余弦相似度得到目标类别。"""
    standardized = torch.tensor([[1.0, 0.0]])
    mean = torch.zeros((1, 2))
    std = torch.ones((1, 2))
    prototypes = torch.eye(2)
    logits = clip_zero_shot_logits(standardized, mean, std, prototypes, 2.0)
    assert logits.argmax(dim=1).item() == 0
    assert math.isclose(float(logits[0, 0]), 2.0)
