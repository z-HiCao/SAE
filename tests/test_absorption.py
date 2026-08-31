import torch

from sae_repro.metrics.absorption import absorption_report


def test_absorption_fixture_detects_parent_contribution() -> None:
    """主 latent 漏检而 child latent 携带 parent 方向时应产生候选。"""
    activations = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    latents = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    decoder = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    coarse = torch.tensor([0, 0, 0, 1, 1])
    fine = torch.tensor([0, 1, 1, 2, 3])
    config = {
        "seed": 1,
        "activation_threshold": 1e-6,
        "splitting_f1_jump": 0.03,
        "max_split_latents": 2,
        "min_decoder_projection": 0.01,
        "min_ablation_effect": 0.01,
        "max_false_negatives_per_parent": 20,
    }
    report = absorption_report(activations, latents, decoder, coarse, fine, config)
    parent_zero = next(row for row in report if row["parent_id"] == 0)
    assert parent_zero["absorption_count"] >= 1

