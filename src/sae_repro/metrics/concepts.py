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


def top_activating_indices(latents: torch.Tensor, top_k: int) -> torch.Tensor:
    """返回每个 latent 的 top activating 样本行号。"""
    count = min(top_k, latents.shape[0])
    return torch.topk(latents, count, dim=0).indices.T


def class_centroid_weights(activations: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    """以每类平均激活构造轻量、可审计的下游概念 probe。"""
    weights = []
    for class_id in range(classes):
        mask = labels == class_id
        if not torch.any(mask):
            weights.append(torch.zeros(activations.shape[1], dtype=activations.dtype))
        else:
            weights.append(activations[mask].mean(dim=0))
    return F.normalize(torch.stack(weights), dim=1)


def latent_target_classes(latents: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    """按各类别平均激活为每个 latent 指派一个候选 fine concept。"""
    means = []
    for class_id in range(classes):
        mask = labels == class_id
        means.append(latents[mask].mean(dim=0))
    return torch.stack(means).argmax(dim=0)

