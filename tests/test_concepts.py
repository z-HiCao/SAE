import numpy as np
import torch

from sae_repro.data.concepts import build_concept_matrix
from sae_repro.metrics.concepts import positive_top_activating_indices


def test_concept_matrix_has_one_fine_and_one_coarse() -> None:
    """每个样本应恰好激活一个 fine 和一个 coarse 概念。"""
    fine = np.asarray([0, 9, 99])
    coarse = np.asarray([0, 1, 19])
    matrix = build_concept_matrix(fine, coarse)
    assert matrix.shape == (3, 120)
    assert np.all(matrix.sum(axis=1) == 2)
    assert matrix[2, 99] == 1
    assert matrix[2, 119] == 1


def test_positive_top_indices_use_negative_one_for_padding() -> None:
    """top image 行号不得用零激活样本伪装成正激活样本。"""
    latents = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    rows, counts = positive_top_activating_indices(latents, top_k=3)
    assert counts.tolist() == [1, 1]
    assert int((rows[0] >= 0).sum()) == 1
    assert int((rows[1] >= 0).sum()) == 1
