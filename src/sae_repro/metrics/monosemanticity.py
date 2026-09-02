from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _minmax_columns(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 latent 做跨样本 min-max 归一化，并标出非常数列。"""
    minimum = activations.min(dim=0).values
    maximum = activations.max(dim=0).values
    span = maximum - minimum
    valid = span > 1e-12
    normalized = torch.zeros_like(activations)
    normalized[:, valid] = (activations[:, valid] - minimum[valid]) / span[valid]
    return normalized, valid


def monosemanticity_scores(
    activations: torch.Tensor,
    semantic_embeddings: torch.Tensor,
) -> torch.Tensor:
    """用不构造 N×N 相似度矩阵的等价公式计算每个 latent 的 MS。"""
    if activations.ndim != 2 or semantic_embeddings.ndim != 2:
        raise ValueError("activations 和 semantic_embeddings 都必须是二维张量")
    if activations.shape[0] != semantic_embeddings.shape[0]:
        raise ValueError("激活和语义嵌入的样本数不一致")
    weights, valid = _minmax_columns(activations.float())
    embeddings = F.normalize(semantic_embeddings.float(), dim=1)
    weighted_sum = embeddings.T @ weights
    weight_square_sum = torch.sum(weights**2, dim=0)
    numerator = 0.5 * (torch.sum(weighted_sum**2, dim=0) - weight_square_sum)
    denominator = 0.5 * (torch.sum(weights, dim=0) ** 2 - weight_square_sum)
    scores = torch.full((activations.shape[1],), float("nan"), dtype=torch.float32)
    usable = valid & (denominator > 1e-12)
    scores[usable] = numerator[usable] / denominator[usable]
    return scores


def monosemanticity_reference(
    activation: torch.Tensor,
    semantic_embeddings: torch.Tensor,
) -> float:
    """只用于小样本测试的显式两两 MS 实现。"""
    values = activation.float()
    span = values.max() - values.min()
    if span <= 1e-12:
        return float("nan")
    weights = (values - values.min()) / span
    embeddings = F.normalize(semantic_embeddings.float(), dim=1)
    similarity = embeddings @ embeddings.T
    relevance = weights[:, None] * weights[None, :]
    upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    denominator = relevance[upper].sum()
    if denominator <= 1e-12:
        return float("nan")
    return float(((relevance[upper] * similarity[upper]).sum() / denominator).cpu())


def positive_support_counts(
    activations: torch.Tensor,
    activation_threshold: float = 0.0,
) -> torch.Tensor:
    """统计每个 SAE latent 超过激活阈值的样本数。"""
    if activations.ndim != 2:
        raise ValueError("activations 必须是二维张量")
    return torch.sum(activations > activation_threshold, dim=0)


def supported_monosemanticity_scores(
    activations: torch.Tensor,
    semantic_embeddings: torch.Tensor,
    minimum_support: int,
    activation_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 MS，并把正激活样本不足的 latent 标记为 NaN。"""
    scores = monosemanticity_scores(activations, semantic_embeddings)
    support = positive_support_counts(activations, activation_threshold)
    scores[support < minimum_support] = float("nan")
    return scores, support


def monosemanticity_subsample_intervals(
    activations: torch.Tensor,
    semantic_embeddings: torch.Tensor,
    latent_ids: list[int],
    repeats: int,
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """通过无放回子采样估计选定 latent 的 MS 稳定区间。"""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("子采样比例必须位于 (0, 1] 区间")
    if repeats <= 0 or not latent_ids:
        return []
    sample_size = max(2, round(len(activations) * fraction))
    selected = activations[:, latent_ids]
    generator = torch.Generator().manual_seed(seed)
    repeated_scores: list[torch.Tensor] = []
    for _ in range(repeats):
        indices = torch.randperm(len(activations), generator=generator)[:sample_size]
        repeated_scores.append(
            monosemanticity_scores(selected[indices], semantic_embeddings[indices])
        )
    matrix = torch.stack(repeated_scores)
    rows: list[dict[str, Any]] = []
    for column, latent_id in enumerate(latent_ids):
        finite = matrix[:, column][torch.isfinite(matrix[:, column])]
        if finite.numel() == 0:
            rows.append(
                {
                    "latent_id": int(latent_id),
                    "finite_repeats": 0,
                    "mean": float("nan"),
                    "lower_95": float("nan"),
                    "upper_95": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "latent_id": int(latent_id),
                "finite_repeats": int(finite.numel()),
                "mean": float(finite.mean()),
                "lower_95": float(torch.quantile(finite, 0.025)),
                "upper_95": float(torch.quantile(finite, 0.975)),
            }
        )
    return rows
