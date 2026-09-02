from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .absorption_controls import (
    benjamini_hochberg,
    child_specificity,
    empirical_null_summary,
    evaluate_candidate_ablation,
    matched_random_candidate_sets,
)
from .concepts import binary_f1
from .splitting import discover_splitting_order, evaluate_splitting_order


@dataclass
class ParentDiscovery:
    """保存仅由发现集确定、可迁移到验证集和测试集的规则。"""

    parent_id: int
    main_latent: int
    direction: torch.Tensor
    projection: torch.Tensor
    split_order: list[int]
    candidates: list[dict[str, Any]]
    discovery_report: dict[str, Any]


def difference_in_means_direction(activations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """用正负样本均值差构造 parent concept 的线性 probe direction。"""
    positive = activations[target].mean(dim=0)
    negative = activations[~target].mean(dim=0)
    return F.normalize((positive - negative).unsqueeze(0), dim=1).squeeze(0)


def stratified_fit_validation_indices(
    labels: torch.Tensor,
    fit_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 coarse 标签分层切分发现集与验证集。"""
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction 必须位于 0 和 1 之间")
    generator = torch.Generator().manual_seed(seed)
    fit_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for label in torch.unique(labels).tolist():
        indices = torch.where(labels == int(label))[0]
        if len(indices) < 2:
            raise ValueError(f"coarse 类别 {label} 少于两个样本，无法分层切分")
        order = indices[torch.randperm(len(indices), generator=generator)]
        fit_count = min(max(1, round(len(order) * fit_fraction)), len(order) - 1)
        fit_parts.append(order[:fit_count])
        validation_parts.append(order[fit_count:])
    return torch.cat(fit_parts), torch.cat(validation_parts)


def discover_absorption_features(
    activations: torch.Tensor,
    latents: torch.Tensor,
    decoder_directions: torch.Tensor,
    coarse_labels: torch.Tensor,
    fine_labels: torch.Tensor,
    config: dict[str, Any],
) -> list[ParentDiscovery]:
    """只在发现集选择 main latent、split 顺序和 absorption 候选。"""
    threshold = float(config["activation_threshold"])
    firing = latents > threshold
    reports: list[ParentDiscovery] = []
    for parent_id in sorted(torch.unique(coarse_labels).tolist()):
        parent = coarse_labels == int(parent_id)
        f1_values = [binary_f1(firing[:, idx], parent) for idx in range(latents.shape[1])]
        main_latent = int(np.argmax(f1_values))
        splitting = discover_splitting_order(
            firing,
            parent,
            int(config["max_split_latents"]),
            float(config["splitting_f1_jump"]),
        )
        direction = difference_in_means_direction(activations, parent)
        projection = decoder_directions @ direction
        false_negative = parent & ~firing[:, main_latent]
        false_count = int(false_negative.sum())
        generator = torch.Generator().manual_seed(int(config["seed"]) + int(parent_id) * 997)
        screened: list[dict[str, Any]] = []
        for latent_id in range(latents.shape[1]):
            if latent_id == main_latent:
                continue
            false_support = int((firing[:, latent_id] & false_negative).sum())
            if false_support < int(config["min_candidate_support"]):
                continue
            active_false = firing[:, latent_id] & false_negative
            mean_effect = float((latents[active_false, latent_id] * projection[latent_id]).mean())
            if float(projection[latent_id]) < float(config["min_decoder_projection"]):
                continue
            if mean_effect < float(config["min_ablation_effect"]):
                continue
            specificity = child_specificity(
                firing[:, latent_id],
                parent,
                fine_labels,
                int(config["child_specificity_null_repeats"]),
                generator,
            )
            total_firing = int(firing[:, latent_id].sum())
            parent_firing = int((firing[:, latent_id] & parent).sum())
            row = {
                "latent_id": latent_id,
                "decoder_projection": float(projection[latent_id]),
                "mean_effect_on_discovery_false_negatives": mean_effect,
                "false_negative_support": false_support,
                "false_negative_coverage": false_support / max(false_count, 1),
                "total_firing_support": total_firing,
                "parent_precision": parent_firing / max(total_firing, 1),
                **specificity,
            }
            row["selection_score"] = float(
                row["false_negative_coverage"] * mean_effect * row["child_lift"]
            )
            row["passes_child_specificity"] = bool(
                row["child_purity"] >= float(config["min_child_purity"])
                and row["child_lift"] >= float(config["min_child_lift"])
            )
            screened.append(row)
        adjusted = benjamini_hochberg(
            [float(row["specificity_p_value"]) for row in screened]
        )
        for row, q_value in zip(screened, adjusted, strict=True):
            row["specificity_q_value"] = q_value
            row["passes_child_specificity"] = bool(
                row["passes_child_specificity"]
                and q_value <= float(config["max_child_specificity_q_value"])
            )
        eligible = [row for row in screened if row["passes_child_specificity"]]
        eligible.sort(key=lambda row: float(row["selection_score"]), reverse=True)
        selected: list[dict[str, Any]] = []
        selected_children: set[int] = set()
        for row in eligible:
            child_id = int(row["child_id"])
            if child_id in selected_children:
                continue
            selected.append(row)
            selected_children.add(child_id)
            if len(selected) >= int(config["max_absorption_candidates"]):
                break
        selected_ids = {int(row["latent_id"]) for row in selected}
        for row in screened:
            row["selected"] = int(row["latent_id"]) in selected_ids
        reports.append(
            ParentDiscovery(
                parent_id=int(parent_id),
                main_latent=main_latent,
                direction=direction,
                projection=projection,
                split_order=[int(value) for value in splitting["latent_order"]],
                candidates=selected,
                discovery_report={
                    "parent_id": int(parent_id),
                    "parent_positive_count": int(parent.sum()),
                    "main_latent": main_latent,
                    "main_latent_f1": float(f1_values[main_latent]),
                    "false_negative_count": false_count,
                    "splitting": splitting,
                    "screened_candidates": screened,
                    "selected_candidate_count": len(selected),
                    "selected_candidate_ids": sorted(selected_ids),
                    "fine_labels_in_parent": sorted(torch.unique(fine_labels[parent]).tolist()),
                },
            )
        )
    return reports


def evaluate_absorption_features(
    discoveries: list[ParentDiscovery],
    discovery_latents: torch.Tensor,
    decoder_directions: torch.Tensor,
    activations: torch.Tensor,
    latents: torch.Tensor,
    coarse_labels: torch.Tensor,
    fine_labels: torch.Tensor,
    config: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    """在不重新选择候选的条件下评价验证集或测试集。"""
    firing = latents > float(config["activation_threshold"])
    output: list[dict[str, Any]] = []
    for discovery in discoveries:
        parent = coarse_labels == discovery.parent_id
        splitting = evaluate_splitting_order(
            firing,
            parent,
            discovery.split_order,
            float(config["splitting_f1_jump"]),
        )
        observed = evaluate_candidate_ablation(
            activations,
            latents,
            decoder_directions,
            discovery.direction,
            parent,
            fine_labels,
            discovery.main_latent,
            discovery.candidates,
            float(config["activation_threshold"]),
            float(config["min_ablation_effect"]),
            int(config["max_false_negatives_per_parent"]),
            int(config["seed"]) + discovery.parent_id * 101,
        )
        selected_ids = [int(row["latent_id"]) for row in discovery.candidates]
        random_sets = matched_random_candidate_sets(
            discovery_latents,
            decoder_directions,
            discovery.projection,
            selected_ids,
            excluded_latents={discovery.main_latent, *selected_ids},
            activation_threshold=float(config["activation_threshold"]),
            repeats=int(config["matched_random_repeats"]),
            pool_size=int(config["matched_random_pool_size"]),
            seed=int(config["seed"]) + discovery.parent_id * 211,
        )
        null_rates: list[float] = []
        for random_ids in random_sets:
            random_candidates = [{"latent_id": value, "child_id": -1} for value in random_ids]
            null_report = evaluate_candidate_ablation(
                activations,
                latents,
                decoder_directions,
                discovery.direction,
                parent,
                fine_labels,
                discovery.main_latent,
                random_candidates,
                float(config["activation_threshold"]),
                float(config["min_ablation_effect"]),
                int(config["max_false_negatives_per_parent"]),
                int(config["seed"]) + discovery.parent_id * 101,
            )
            null_rates.append(float(null_report["absorption_rate_over_tested_false_negatives"]))
        observed_rate = float(observed["absorption_rate_over_tested_false_negatives"])
        output.append(
            {
                "split": split_name,
                "parent_id": discovery.parent_id,
                "main_latent": discovery.main_latent,
                "candidate_ids": selected_ids,
                "candidate_count": len(selected_ids),
                "splitting": splitting,
                **observed,
                "matched_random_control": {
                    "repeats": len(null_rates),
                    **empirical_null_summary(observed_rate, null_rates),
                },
            }
        )
    return output


def absorption_report(
    activations: torch.Tensor,
    latents: torch.Tensor,
    decoder_directions: torch.Tensor,
    coarse_labels: torch.Tensor,
    fine_labels: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """兼容旧接口，但内部强制分层切分发现集和留出评价集。"""
    fit, evaluation = stratified_fit_validation_indices(
        coarse_labels,
        float(config.get("discovery_fit_fraction", 0.7)),
        int(config["seed"]),
    )
    discoveries = discover_absorption_features(
        activations[fit],
        latents[fit],
        decoder_directions,
        coarse_labels[fit],
        fine_labels[fit],
        config,
    )
    return evaluate_absorption_features(
        discoveries,
        latents[fit],
        decoder_directions,
        activations[evaluation],
        latents[evaluation],
        coarse_labels[evaluation],
        fine_labels[evaluation],
        config,
        "heldout",
    )
