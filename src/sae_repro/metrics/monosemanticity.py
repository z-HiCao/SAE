from __future__ import annotations

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

