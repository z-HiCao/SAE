from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .concepts import binary_f1


def difference_in_means_direction(activations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """用正负样本均值差构造 parent concept 的线性 probe direction。"""
    positive = activations[target].mean(dim=0)
    negative = activations[~target].mean(dim=0)
    return F.normalize((positive - negative).unsqueeze(0), dim=1).squeeze(0)


def _splitting_count(
    firing: torch.Tensor,
    target: torch.Tensor,
    max_latents: int,
    f1_jump: float,
) -> tuple[int, list[int], list[float]]:
    """按单 latent F1 排序，并统计并集 F1 的显著跳升次数。"""
    individual = [binary_f1(firing[:, idx], target) for idx in range(firing.shape[1])]
    order = np.argsort(individual)[::-1][:max_latents].tolist()
    combined = torch.zeros_like(target, dtype=torch.bool)
    curve = []
    splits = 0
    previous = 0.0
    for latent_id in order:
        combined |= firing[:, latent_id]
        current = binary_f1(combined, target)
        curve.append(current)
        if current - previous > f1_jump:
            splits += 1
        previous = current
    return splits, order, curve


def absorption_report(
    activations: torch.Tensor,
    latents: torch.Tensor,
    decoder_directions: torch.Tensor,
    coarse_labels: torch.Tensor,
    fine_labels: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """为每个 coarse parent 估计 splitting 和保守 absorption 候选。"""
    threshold = float(config["activation_threshold"])
    firing = latents > threshold
    reports: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(int(config["seed"]))
    for parent_id in sorted(torch.unique(coarse_labels).tolist()):
        parent = coarse_labels == int(parent_id)
        f1_values = [binary_f1(firing[:, idx], parent) for idx in range(latents.shape[1])]
        main_latent = int(np.argmax(f1_values))
        splits, split_latents, f1_curve = _splitting_count(
            firing,
            parent,
            int(config["max_split_latents"]),
            float(config["splitting_f1_jump"]),
        )
        false_negative = parent & ~firing[:, main_latent]
        false_indices = torch.where(false_negative)[0]
        maximum = int(config["max_false_negatives_per_parent"])
        if len(false_indices) > maximum:
            selection = torch.randperm(len(false_indices), generator=generator)[:maximum]
            false_indices = false_indices[selection]

        direction = difference_in_means_direction(activations, parent)
        projection = decoder_directions @ direction
        projection[main_latent] = 0.0
        absorption_count = 0
        candidate_counts: dict[int, int] = {}
        effects = []
        for row in false_indices.tolist():
            contributions = latents[row] * projection
            candidate = int(torch.argmax(contributions))
            effect = float(contributions[candidate])
            if (
                float(projection[candidate]) >= float(config["min_decoder_projection"])
                and effect >= float(config["min_ablation_effect"])
                and firing[row, candidate]
            ):
                absorption_count += 1
                candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
                effects.append(effect)
        denominator = max(int(parent.sum()), 1)
        reports.append(
            {
                "parent_id": int(parent_id),
                "main_latent": main_latent,
                "main_latent_f1": float(f1_values[main_latent]),
                "splitting_count": int(splits),
                "split_latents": split_latents,
                "split_f1_curve": f1_curve,
                "false_negative_count": int(false_negative.sum()),
                "tested_false_negatives": int(len(false_indices)),
                "absorption_count": int(absorption_count),
                "absorption_rate_over_parent_positives": float(absorption_count / denominator),
                "mean_ablation_effect": float(np.mean(effects)) if effects else 0.0,
                "candidate_latent_counts": {str(key): value for key, value in candidate_counts.items()},
                "fine_labels_in_parent": sorted(torch.unique(fine_labels[parent]).tolist()),
            }
        )
    return reports

