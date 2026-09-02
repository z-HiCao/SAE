import torch

from sae_repro.metrics.absorption import (
    discover_absorption_features,
    evaluate_absorption_features,
)


def test_absorption_fixture_detects_parent_contribution() -> None:
    """固定 child 候选在留出评价中应产生可测的 decoder 消融效应。"""
    activations = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    latents = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ]
    )
    decoder = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coarse = torch.tensor([0] * 8 + [1] * 4)
    fine = torch.tensor([0] * 4 + [1] * 4 + [2] * 2 + [3] * 2)
    config = {
        "seed": 1,
        "activation_threshold": 1e-6,
        "splitting_f1_jump": 0.03,
        "max_split_latents": 3,
        "min_decoder_projection": 0.01,
        "min_ablation_effect": 0.01,
        "max_false_negatives_per_parent": 20,
        "min_candidate_support": 1,
        "min_child_purity": 0.5,
        "min_child_lift": 1.0,
        "max_child_specificity_q_value": 1.0,
        "max_absorption_candidates": 2,
        "child_specificity_null_repeats": 10,
        "matched_random_repeats": 5,
        "matched_random_pool_size": 2,
    }
    discoveries = discover_absorption_features(
        activations, latents, decoder, coarse, fine, config
    )
    report = evaluate_absorption_features(
        discoveries,
        latents,
        decoder,
        activations,
        latents,
        coarse,
        fine,
        config,
        "test",
    )
    parent_zero = next(row for row in report if row["parent_id"] == 0)
    assert parent_zero["absorption_count"] >= 1
    assert parent_zero["absorption_with_matching_child_count"] >= 1
    assert parent_zero["candidate_ids"] == [1]
