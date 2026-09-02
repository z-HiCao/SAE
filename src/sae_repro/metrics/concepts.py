from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """计算二分类 F1，空预测或空目标时返回 0。"""
    prediction = prediction.bool()
    target = target.bool()
    true_positive = torch.sum(prediction & target).float()
    false_positive = torch.sum(prediction & ~target).float()
    false_negative = torch.sum(~prediction & target).float()
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator <= 0:
        return 0.0
    return float((2 * true_positive / denominator).cpu())


def best_latent_per_concept(
    latents: torch.Tensor,
    concepts: torch.Tensor,
    activation_threshold: float,
) -> list[dict[str, Any]]:
    """为每个已知概念寻找 F1 最高的单个 SAE latent。"""
    firing = latents > activation_threshold
    rows: list[dict[str, Any]] = []
    for concept_id in range(concepts.shape[1]):
        target = concepts[:, concept_id] > 0
        scores = [binary_f1(firing[:, latent_id], target) for latent_id in range(latents.shape[1])]
        best = int(np.argmax(scores))
        rows.append({"concept_id": concept_id, "latent_id": best, "f1": float(scores[best])})
    return rows


def dictionary_match_scores(
    decoder_directions: torch.Tensor,
    feature_directions: torch.Tensor,
) -> torch.Tensor:
    """计算每个真实 feature direction 与最相似 decoder direction 的 cosine。"""
    decoder = F.normalize(decoder_directions.float(), dim=1)
    features = F.normalize(feature_directions.float(), dim=1)
    similarities = features @ decoder.T
    return similarities.max(dim=1).values


def positive_top_activating_indices(
    latents: torch.Tensor,
    top_k: int,
    activation_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只返回真正正激活的 top 样本，不足 top_k 的位置使用 -1。"""
    count = min(top_k, latents.shape[0])
    values, indices = torch.topk(latents, count, dim=0)
    rows = indices.T.contiguous()
    positive = values.T > activation_threshold
    rows[~positive] = -1
    return rows, positive.sum(dim=1)
