import math

import torch

from sae_repro.metrics.monosemanticity import (
    monosemanticity_reference,
    monosemanticity_scores,
    supported_monosemanticity_scores,
)


def test_ms_optimized_matches_pairwise_reference() -> None:
    """低内存 MS 必须与显式两两公式一致。"""
    generator = torch.Generator().manual_seed(7)
    activations = torch.rand((12, 5), generator=generator)
    embeddings = torch.rand((12, 4), generator=generator)
    optimized = monosemanticity_scores(activations, embeddings)
    for latent_id in range(activations.shape[1]):
        reference = monosemanticity_reference(activations[:, latent_id], embeddings)
        assert math.isclose(float(optimized[latent_id]), reference, rel_tol=1e-5, abs_tol=1e-5)


def test_ms_support_filter_rejects_tiny_support() -> None:
    """只有一个正激活样本的 latent 不应被当成可靠单义特征。"""
    activations = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    embeddings = torch.eye(4)
    scores, support = supported_monosemanticity_scores(
        activations,
        embeddings,
        minimum_support=2,
    )
    assert support.tolist() == [1, 2]
    assert math.isnan(float(scores[0]))
    assert torch.isfinite(scores[1])
