import torch

from sae_repro.metrics.cross_model_alignment import (
    alignment_null_controls,
    bootstrap_alignment_intervals,
    latent_alignment_report,
)
from sae_repro.metrics.universality import (
    cross_reconstruction_matrix,
    shuffled_target_cross_reconstruction_matrix,
)
from sae_repro.sae.models import UniversalSAE


def test_identical_latents_have_strong_cross_model_alignment() -> None:
    """完全一致的 latent 应得到高激活和语义对齐分数。"""
    first = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    second = first.clone()
    fine = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    coarse = fine.clone()
    rows, summary = latent_alignment_report(
        first,
        second,
        fine,
        coarse,
        activation_threshold=1e-6,
        top_k=2,
        minimum_support=2,
    )
    assert summary["mean_activation_pearson"] > 0.99
    assert summary["mean_top_image_jaccard"] == 1.0
    assert summary["fine_label_agreement_rate"] == 1.0

    controls = alignment_null_controls(
        first,
        second,
        fine,
        coarse,
        rows,
        summary,
        activation_threshold=1e-6,
        repeats=20,
        seed=3,
    )
    assert controls["sample_permutation"]["mean_activation_pearson"]["null_mean"] < 1.0
    intervals = bootstrap_alignment_intervals(rows, repeats=20, confidence=0.9, seed=4)
    assert "activation_pearson" in intervals


def test_shuffled_target_reduces_cross_reconstruction() -> None:
    """完美配对重构应优于目标样本被打乱后的负对照。"""
    model = UniversalSAE([2, 2], latent_dim=2, k=2)
    with torch.no_grad():
        for encoder in model.encoders:
            encoder.weight.copy_(torch.eye(2))
            encoder.bias.zero_()
        for decoder in model.decoders:
            decoder.weight.copy_(torch.eye(2))
            decoder.bias.zero_()
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    paired = cross_reconstruction_matrix(model, [values, values], torch.device("cpu"), 2)
    shuffled = shuffled_target_cross_reconstruction_matrix(
        model,
        [values, values],
        torch.tensor([1, 0, 3, 2]),
        torch.device("cpu"),
        2,
    )
    assert torch.all(paired > shuffled)
