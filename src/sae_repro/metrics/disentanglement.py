from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


def _binary_statistics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    """返回二分类的计数、precision、recall 和 F1。"""
    prediction = prediction.bool()
    target = target.bool()
    true_positive = int(torch.sum(prediction & target))
    false_positive = int(torch.sum(prediction & ~target))
    false_negative = int(torch.sum(~prediction & target))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    denominator = precision + recall
    f1 = 0.0 if denominator == 0 else 2.0 * precision * recall / denominator
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _candidate_thresholds(values: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """只用拟合集的正值构造阈值，避免在测试集上调参。"""
    positive = values[torch.isfinite(values) & (values > 0)]
    if positive.numel() == 0:
        return torch.empty(0, dtype=values.dtype)
    requested = torch.tensor(quantiles, dtype=positive.dtype, device=positive.device)
    thresholds = torch.quantile(positive, requested)
    minimum = positive.min() - torch.finfo(positive.dtype).eps
    return torch.unique(torch.cat((minimum.reshape(1), thresholds))).cpu()


def select_unit_concept_mapping(
    calibration_activations: torch.Tensor,
    validation_activations: torch.Tensor,
    validation_concepts: torch.Tensor,
    quantiles: list[float],
    allow_negative_direction: bool,
) -> list[dict[str, Any]]:
    """在验证集上为每个概念选择单个单元、方向和激活阈值。"""
    if calibration_activations.shape[1] != validation_activations.shape[1]:
        raise ValueError("校准集和验证集的单元维度不一致")
    signs = (1.0, -1.0) if allow_negative_direction else (1.0,)
    thresholds: dict[tuple[int, float], torch.Tensor] = {}
    for unit_id in range(calibration_activations.shape[1]):
        for sign in signs:
            thresholds[(unit_id, sign)] = _candidate_thresholds(
                calibration_activations[:, unit_id] * sign,
                quantiles,
            )

    rows: list[dict[str, Any]] = []
    for concept_id in range(validation_concepts.shape[1]):
        target = validation_concepts[:, concept_id] > 0
        best: dict[str, Any] = {
            "concept_id": int(concept_id),
            "unit_id": -1,
            "sign": 1.0,
            "threshold": 0.0,
            "validation_f1": 0.0,
        }
        for unit_id in range(validation_activations.shape[1]):
            for sign in signs:
                oriented = validation_activations[:, unit_id] * sign
                for threshold in thresholds[(unit_id, sign)].tolist():
                    statistics = _binary_statistics(oriented > threshold, target)
                    if float(statistics["f1"]) > float(best["validation_f1"]):
                        best = {
                            "concept_id": int(concept_id),
                            "unit_id": int(unit_id),
                            "sign": float(sign),
                            "threshold": float(threshold),
                            "validation_f1": float(statistics["f1"]),
                        }
        rows.append(best)
    return rows


def evaluate_unit_concept_mapping(
    activations: torch.Tensor,
    concepts: torch.Tensor,
    mapping: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """在独立测试集上评价验证阶段固定的单元—概念映射。"""
    rows: list[dict[str, Any]] = []
    for selected in mapping:
        concept_id = int(selected["concept_id"])
        unit_id = int(selected["unit_id"])
        sign = float(selected["sign"])
        threshold = float(selected["threshold"])
        if unit_id < 0:
            statistics = _binary_statistics(
                torch.zeros(len(activations), dtype=torch.bool),
                concepts[:, concept_id] > 0,
            )
        else:
            prediction = activations[:, unit_id] * sign > threshold
            statistics = _binary_statistics(prediction, concepts[:, concept_id] > 0)
        rows.append({**selected, **statistics})
    return rows


def mapping_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """汇总概念映射，并统计不同概念复用同一单元的程度。"""
    if not rows:
        return {
            "mean_f1": 0.0,
            "median_f1": 0.0,
            "concepts_f1_ge_0_5": 0,
            "unique_selected_units": 0,
        }
    scores = torch.tensor([float(row["f1"]) for row in rows])
    units = {int(row["unit_id"]) for row in rows if int(row["unit_id"]) >= 0}
    return {
        "mean_f1": float(scores.mean()),
        "median_f1": float(scores.median()),
        "concepts_f1_ge_0_3": int(torch.sum(scores >= 0.3)),
        "concepts_f1_ge_0_5": int(torch.sum(scores >= 0.5)),
        "concepts_f1_ge_0_8": int(torch.sum(scores >= 0.8)),
        "unique_selected_units": len(units),
    }


def latent_purity_rows(
    latents: torch.Tensor,
    fine_labels: torch.Tensor,
    coarse_labels: torch.Tensor,
    activation_threshold: float,
) -> list[dict[str, float | int]]:
    """从 latent 出发计算激活样本的 fine/coarse 纯度与归一化熵。"""
    rows: list[dict[str, float | int]] = []
    fine_classes = int(fine_labels.max()) + 1
    coarse_classes = int(coarse_labels.max()) + 1
    for latent_id in range(latents.shape[1]):
        active = latents[:, latent_id] > activation_threshold
        support = int(active.sum())
        if support == 0:
            rows.append(
                {
                    "latent_id": latent_id,
                    "support": 0,
                    "best_fine_class": -1,
                    "fine_purity": 0.0,
                    "fine_entropy": 0.0,
                    "best_coarse_class": -1,
                    "coarse_purity": 0.0,
                    "coarse_entropy": 0.0,
                }
            )
            continue
        fine_counts = torch.bincount(fine_labels[active], minlength=fine_classes).float()
        coarse_counts = torch.bincount(coarse_labels[active], minlength=coarse_classes).float()
        fine_probabilities = fine_counts / support
        coarse_probabilities = coarse_counts / support
        positive_fine = fine_probabilities[fine_probabilities > 0]
        positive_coarse = coarse_probabilities[coarse_probabilities > 0]
        fine_entropy = -torch.sum(positive_fine * torch.log(positive_fine)) / math.log(fine_classes)
        coarse_entropy = -torch.sum(positive_coarse * torch.log(positive_coarse)) / math.log(
            coarse_classes
        )
        rows.append(
            {
                "latent_id": latent_id,
                "support": support,
                "best_fine_class": int(torch.argmax(fine_counts)),
                "fine_purity": float(fine_probabilities.max()),
                "fine_entropy": float(fine_entropy),
                "best_coarse_class": int(torch.argmax(coarse_counts)),
                "coarse_purity": float(coarse_probabilities.max()),
                "coarse_entropy": float(coarse_entropy),
            }
        )
    return rows


def one_to_one_dictionary_match(
    decoder_directions: torch.Tensor,
    feature_directions: torch.Tensor,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """用 Hungarian matching 评价 ground-truth feature 与 decoder 的一对一恢复。"""
    from scipy.optimize import linear_sum_assignment

    decoder = F.normalize(decoder_directions.float(), dim=1)
    features = F.normalize(feature_directions.float(), dim=1)
    similarities = features @ decoder.T
    feature_ids, latent_ids = linear_sum_assignment((-similarities).cpu().numpy())
    rows = [
        {
            "concept_id": int(feature_id),
            "latent_id": int(latent_id),
            "cosine": float(similarities[feature_id, latent_id]),
        }
        for feature_id, latent_id in zip(feature_ids.tolist(), latent_ids.tolist())
    ]
    scores = torch.tensor([float(row["cosine"]) for row in rows])
    summary: dict[str, float | int] = {
        "matched_features": len(rows),
        "mean_cosine": float(scores.mean()) if len(scores) else 0.0,
        "median_cosine": float(scores.median()) if len(scores) else 0.0,
        "matches_cosine_ge_0_5": int(torch.sum(scores >= 0.5)),
        "matches_cosine_ge_0_8": int(torch.sum(scores >= 0.8)),
        "matches_cosine_ge_0_9": int(torch.sum(scores >= 0.9)),
    }
    return rows, summary


@torch.no_grad()
def toy_concept_ablation(
    sae_model: torch.nn.Module,
    test_latents: torch.Tensor,
    test_concepts: torch.Tensor,
    concept_mapping: list[dict[str, Any]],
    p01_weight: torch.Tensor,
    p01_bias: torch.Tensor,
    activation_threshold: float,
) -> list[dict[str, float | int]]:
    """消融单个 SAE latent，并经 P01 decoder 测量目标概念输出变化。"""
    sae_model = sae_model.to("cpu").eval()
    decoder = sae_model.decoder_directions().detach().cpu()
    baseline_hidden = sae_model.decode(test_latents).detach().cpu()
    baseline_concepts = F.relu(baseline_hidden @ p01_weight + p01_bias)
    rows: list[dict[str, float | int]] = []
    for selected in concept_mapping:
        concept_id = int(selected["concept_id"])
        latent_id = int(selected["unit_id"])
        if latent_id < 0:
            continue
        selected_threshold = max(activation_threshold, float(selected.get("threshold", 0.0)))
        active_target = (test_concepts[:, concept_id] > 0) & (
            test_latents[:, latent_id] > selected_threshold
        )
        support = int(active_target.sum())
        if support == 0:
            rows.append(
                {
                    "concept_id": concept_id,
                    "latent_id": latent_id,
                    "active_target_support": 0,
                    "mean_target_output_drop": 0.0,
                    "mean_off_target_absolute_change": 0.0,
                    "specificity_ratio": 0.0,
                }
            )
            continue
        contribution = test_latents[:, latent_id : latent_id + 1] * decoder[latent_id]
        ablated_hidden = baseline_hidden - contribution
        ablated_concepts = F.relu(ablated_hidden @ p01_weight + p01_bias)
        change = baseline_concepts[active_target] - ablated_concepts[active_target]
        target_drop = float(change[:, concept_id].mean())
        off_target_mask = torch.ones(change.shape[1], dtype=torch.bool)
        off_target_mask[concept_id] = False
        off_target = float(change[:, off_target_mask].abs().mean())
        rows.append(
            {
                "concept_id": concept_id,
                "latent_id": latent_id,
                "active_target_support": support,
                "mean_target_output_drop": target_drop,
                "mean_off_target_absolute_change": off_target,
                "specificity_ratio": target_drop / max(off_target, 1e-12),
            }
        )
    return rows
