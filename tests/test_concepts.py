import numpy as np

from sae_repro.data.concepts import build_concept_matrix


def test_concept_matrix_has_one_fine_and_one_coarse() -> None:
    """每个样本应恰好激活一个 fine 和一个 coarse 概念。"""
    fine = np.asarray([0, 9, 99])
    coarse = np.asarray([0, 1, 19])
    matrix = build_concept_matrix(fine, coarse)
    assert matrix.shape == (3, 120)
    assert np.all(matrix.sum(axis=1) == 2)
    assert matrix[2, 99] == 1
    assert matrix[2, 119] == 1

