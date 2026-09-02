from __future__ import annotations

from typing import Any

import numpy as np
import torch


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """对同一父概念内的候选执行 Benjamini-Hochberg FDR 校正。"""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    total = len(values)
    for reverse_rank in range(total - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * total / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def child_specificity(
    candidate_firing: torch.Tensor,
    parent: torch.Tensor,
    fine_labels: torch.Tensor,
    null_repeats: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """评价候选 latent 在父类内部是否偏向某个 fine 子类。"""
    parent_labels = fine_labels[parent]
    selected = candidate_firing[parent]
    selected_labels = parent_labels[selected]
    if selected_labels.numel() == 0:
        return {
            "child_id": -1,
            "child_support": 0,
            "child_purity": 0.0,
            "child_lift": 0.0,
            "specificity_p_value": 1.0,
        }
    class_count = int(torch.max(fine_labels)) + 1
    counts = torch.bincount(selected_labels.long(), minlength=class_count).float()
    parent_counts = torch.bincount(parent_labels.long(), minlength=class_count).float()
    child_id = int(torch.argmax(counts))
    purity = float(counts[child_id] / counts.sum().clamp_min(1.0))
    prior = float(parent_counts[child_id] / parent_counts.sum().clamp_min(1.0))
    lift = purity / max(prior, 1e-12)

    null_values: list[float] = []
    for _ in range(null_repeats):
        permutation = torch.randperm(len(parent_labels), generator=generator)
        shuffled = parent_labels[permutation][selected]
        shuffled_counts = torch.bincount(shuffled.long(), minlength=class_count).float()
        null_values.append(float(shuffled_counts.max() / shuffled_counts.sum().clamp_min(1.0)))
    p_value = (
        float((1 + sum(value >= purity for value in null_values)) / (len(null_values) + 1))
        if null_values
        else 1.0
    )
    return {
        "child_id": child_id,
        "child_support": int(counts[child_id]),
        "child_purity": purity,
        "child_lift": float(lift),
        "specificity_p_value": p_value,
    }


def matched_random_candidate_sets(
    discovery_latents: torch.Tensor,
    decoder_directions: torch.Tensor,
    projection: torch.Tensor,
    selected_latents: list[int],
    excluded_latents: set[int],
    activation_threshold: float,
    repeats: int,
    pool_size: int,
    seed: int,
) -> list[list[int]]:
    """按 firing rate、decoder norm 和 parent projection 匹配随机候选集合。"""
    if not selected_latents or repeats <= 0:
        return []
    firing_rate = (discovery_latents > activation_threshold).float().mean(dim=0)
    decoder_norm = torch.linalg.norm(decoder_directions.float(), dim=1)
    features = torch.stack([firing_rate, decoder_norm, projection.float()], dim=1)
    scale = features.std(dim=0).clamp_min(1e-6)
    available = [
        index for index in range(discovery_latents.shape[1]) if index not in excluded_latents
    ]
    if not available:
        return []
    generator = torch.Generator().manual_seed(seed)
    output: list[list[int]] = []
    for _ in range(repeats):
        chosen: list[int] = []
        for selected in selected_latents:
            pool = [index for index in available if index not in chosen]
            if not pool:
                break
            distances = torch.linalg.norm(
                (features[pool] - features[selected].unsqueeze(0)) / scale,
                dim=1,
            )
            nearest_count = min(max(1, pool_size), len(pool))
            nearest = torch.topk(distances, nearest_count, largest=False).indices
            pick = int(torch.randint(nearest_count, (1,), generator=generator))
            chosen.append(pool[int(nearest[pick])])
        if len(chosen) == len(selected_latents):
            output.append(chosen)
    return output


def evaluate_candidate_ablation(
    activations: torch.Tensor,
    latents: torch.Tensor,
    decoder_directions: torch.Tensor,
    direction: torch.Tensor,
    parent: torch.Tensor,
    fine_labels: torch.Tensor,
    main_latent: int,
    candidates: list[dict[str, Any]],
    activation_threshold: float,
    minimum_effect: float,
    maximum_false_negatives: int,
    seed: int,
) -> dict[str, Any]:
    """在固定候选集合上执行 decoder 重建消融，并评价独立数据。"""
    firing = latents > activation_threshold
    false_negative = parent & ~firing[:, main_latent]
    false_indices = torch.where(false_negative)[0]
    generator = torch.Generator().manual_seed(seed)
    if len(false_indices) > maximum_false_negatives:
        order = torch.randperm(len(false_indices), generator=generator)[:maximum_false_negatives]
        false_indices = false_indices[order]
    candidate_ids = [int(row["latent_id"]) for row in candidates]
    parent_count = int(parent.sum())
    if len(false_indices) == 0 or not candidate_ids:
        return {
            "parent_positive_count": parent_count,
            "false_negative_count": int(false_negative.sum()),
            "tested_false_negatives": int(len(false_indices)),
            "absorption_count": 0,
            "absorption_with_matching_child_count": 0,
            "absorption_rate_over_tested_false_negatives": 0.0,
            "absorption_rate_over_parent_positives": 0.0,
            "matching_child_rate_over_absorbed": 0.0,
            "mean_decoder_ablation_effect": 0.0,
            "parent_probe_score_correlation": float("nan"),
        }

    selected_latents = latents[false_indices]
    reconstructed = selected_latents @ decoder_directions
    ablated_latents = selected_latents.clone()
    ablated_latents[:, candidate_ids] = 0.0
    ablated = ablated_latents @ decoder_directions
    effects = (reconstructed - ablated) @ direction
    candidate_active = firing[false_indices][:, candidate_ids]
    any_active = candidate_active.any(dim=1)
    absorbed = any_active & (effects >= minimum_effect)

    child_ids = torch.tensor([int(row["child_id"]) for row in candidates])
    sample_children = fine_labels[false_indices].unsqueeze(1)
    matching_child = (candidate_active & (sample_children == child_ids.unsqueeze(0))).any(dim=1)
    absorbed_with_matching_child = absorbed & matching_child

    original_scores = activations @ direction
    reconstructed_scores = (latents @ decoder_directions) @ direction
    centered_original = original_scores - original_scores.mean()
    centered_reconstructed = reconstructed_scores - reconstructed_scores.mean()
    denominator = torch.linalg.norm(centered_original) * torch.linalg.norm(centered_reconstructed)
    correlation = (
        float(torch.dot(centered_original, centered_reconstructed) / denominator)
        if float(denominator) > 1e-12
        else float("nan")
    )
    absorption_count = int(absorbed.sum())
    matching_count = int(absorbed_with_matching_child.sum())
    return {
        "parent_positive_count": parent_count,
        "false_negative_count": int(false_negative.sum()),
        "tested_false_negatives": int(len(false_indices)),
        "absorption_count": absorption_count,
        "absorption_with_matching_child_count": matching_count,
        "absorption_rate_over_tested_false_negatives": absorption_count
        / max(len(false_indices), 1),
        "absorption_rate_over_parent_positives": absorption_count / max(parent_count, 1),
        "matching_child_rate_over_absorbed": matching_count / max(absorption_count, 1),
        "mean_decoder_ablation_effect": float(effects[absorbed].mean())
        if bool(absorbed.any())
        else 0.0,
        "parent_probe_score_correlation": correlation,
    }


def empirical_null_summary(observed: float, values: list[float]) -> dict[str, float]:
    """汇总匹配随机候选的零假设分布。"""
    if not values:
        return {
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "empirical_p_value": float("nan"),
        }
    array = np.asarray(values, dtype=float)
    return {
        "null_mean": float(array.mean()),
        "null_std": float(array.std()),
        "empirical_p_value": float((1 + np.sum(array >= observed)) / (len(array) + 1)),
    }
