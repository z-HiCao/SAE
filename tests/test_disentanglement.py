import math

import torch

from sae_repro.metrics.disentanglement import (
    evaluate_unit_concept_mapping,
    one_to_one_dictionary_match,
    select_unit_concept_mapping,
)


def test_unit_mapping_uses_validation_then_generalizes_to_test() -> None:
    """验证阶段选择的单元和方向应能原样用于独立测试集。"""
    calibration = torch.tensor([[0.0, -2.0], [1.0, -1.0], [2.0, 0.0], [3.0, 1.0]])
    validation = calibration.clone()
    concepts = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    mapping = select_unit_concept_mapping(
        calibration,
        validation,
        concepts,
        quantiles=[0.25],
        allow_negative_direction=True,
    )
    result = evaluate_unit_concept_mapping(validation, concepts, mapping)
    assert result[0]["unit_id"] in {0, 1}
    assert math.isclose(float(result[0]["f1"]), 1.0)


def test_hungarian_match_does_not_reuse_latent() -> None:
    """一对一方向匹配不能把同一个 SAE latent 分给多个真实 feature。"""
    features = torch.eye(3)
    decoder = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    rows, summary = one_to_one_dictionary_match(decoder, features)
    assert len({int(row["latent_id"]) for row in rows}) == 3
    assert summary["matches_cosine_ge_0_9"] == 3
