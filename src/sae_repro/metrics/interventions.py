from __future__ import annotations

from typing import Any

import torch

from sae_repro.models.clip_zero_shot import clip_zero_shot_logits


def positive_quantile(values: torch.Tensor, quantile: float) -> float:
    """只在正激活值中计算分位数，避免稀疏 latent 得到零钳制值。"""
    positive = values[torch.isfinite(values) & (values > 0)]
    if positive.numel() == 0:
        return 0.0
    return float(torch.quantile(positive.float(), quantile))


def _target_classes(
    train_latents: torch.Tensor,
    train_labels: torch.Tensor,
    classes: int,
) -> torch.Tensor:
    """按照各类别平均激活为每个 latent 指派候选目标类别。"""
    means = []
    for class_id in range(classes):
        mask = train_labels == class_id
        if torch.any(mask):
            means.append(train_latents[mask].mean(dim=0))
        else:
            means.append(torch.zeros(train_latents.shape[1]))
    return torch.stack(means).argmax(dim=0)


def _matched_control_latent(
    latent_id: int,
    support: torch.Tensor,
    eligible: torch.Tensor,
    target_classes: torch.Tensor,
) -> int:
    """选择激活频率接近且目标类别不同的 latent 作为干预对照。"""
    candidates = torch.where(eligible)[0]
    candidates = candidates[
        (candidates != latent_id) & (target_classes[candidates] != target_classes[latent_id])
    ]
    if candidates.numel() == 0:
        return -1
    distance = torch.abs(support[candidates].float() - support[latent_id].float())
    return int(candidates[torch.argmin(distance)])


def _off_target_absolute_change(delta: torch.Tensor, target: int) -> float:
    """计算除目标类别外的平均绝对 logit 变化。"""
    mask = torch.ones(delta.shape[1], dtype=torch.bool)
    mask[target] = False
    return float(delta[:, mask].abs().mean())


@torch.no_grad()
def evaluate_clip_interventions(
    train_latents: torch.Tensor,
    test_latents: torch.Tensor,
    test_reconstruction: torch.Tensor,
    decoder_directions: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    ms_scores: torch.Tensor,
    activation_mean: torch.Tensor,
    activation_std: torch.Tensor,
    text_prototypes: torch.Tensor,
    logit_scale: float,
    intervention_latents: int,
    minimum_support: int,
    quantile: float,
    doses: list[float],
    activation_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """评价 SAE latent 的注入、消融、剂量响应和频率匹配对照。"""
    if not doses:
        raise ValueError("至少需要一个干预剂量")
    classes = text_prototypes.shape[0]
    targets = _target_classes(train_latents, train_labels, classes)
    train_support = torch.sum(train_latents > activation_threshold, dim=0)
    test_support = torch.sum(test_latents > activation_threshold, dim=0)
    eligible = (
        torch.isfinite(ms_scores)
        & (train_support >= minimum_support)
        & (test_support >= minimum_support)
    )
    valid = torch.where(eligible)[0]
    if valid.numel() == 0:
        return [], {
            "eligible_latents": 0,
            "evaluated_latents": 0,
            "positive_injection_effects": 0,
            "negative_ablation_effects": 0,
        }
    order = valid[torch.argsort(ms_scores[valid], descending=True)]
    order = order[:intervention_latents]
    dose_values = sorted(float(value) for value in doses)
    baseline_logits = clip_zero_shot_logits(
        test_reconstruction,
        activation_mean,
        activation_std,
        text_prototypes,
        logit_scale,
    )
    rows: list[dict[str, Any]] = []
    for latent_id in order.tolist():
        target = int(targets[latent_id])
        clamp = positive_quantile(train_latents[:, latent_id], quantile)
        control_id = _matched_control_latent(latent_id, train_support, eligible, targets)
        control_clamp = (
            positive_quantile(train_latents[:, control_id], quantile) if control_id >= 0 else 0.0
        )
        injection_mask = (test_latents[:, latent_id] <= activation_threshold) & (
            test_labels != target
        )
        injection_indices = torch.where(injection_mask)[0]
        dose_rows: list[dict[str, float]] = []
        for dose in dose_values:
            if injection_indices.numel() == 0:
                dose_rows.append(
                    {
                        "dose": float(dose),
                        "mean_target_logit_delta": 0.0,
                        "mean_off_target_absolute_delta": 0.0,
                        "target_top1_rate_before": 0.0,
                        "target_top1_rate_after": 0.0,
                        "matched_control_target_delta": 0.0,
                    }
                )
                continue
            original_value = test_latents[injection_indices, latent_id]
            coefficient_delta = float(dose) * clamp - original_value
            modified = test_reconstruction[injection_indices] + coefficient_delta[:, None] * decoder_directions[
                latent_id
            ]
            modified_logits = clip_zero_shot_logits(
                modified,
                activation_mean,
                activation_std,
                text_prototypes,
                logit_scale,
            )
            before = baseline_logits[injection_indices]
            delta = modified_logits - before
            control_delta = 0.0
            if control_id >= 0:
                control_original = test_latents[injection_indices, control_id]
                control_target = torch.full_like(control_original, float(dose) * control_clamp)
                control_new = torch.maximum(control_original, control_target)
                control_coefficient = control_new - control_original
                control_modified = (
                    test_reconstruction[injection_indices]
                    + control_coefficient[:, None] * decoder_directions[control_id]
                )
                control_logits = clip_zero_shot_logits(
                    control_modified,
                    activation_mean,
                    activation_std,
                    text_prototypes,
                    logit_scale,
                )
                control_delta = float((control_logits[:, target] - before[:, target]).mean())
            dose_rows.append(
                {
                    "dose": float(dose),
                    "mean_target_logit_delta": float(delta[:, target].mean()),
                    "mean_off_target_absolute_delta": _off_target_absolute_change(delta, target),
                    "target_top1_rate_before": float((before.argmax(dim=1) == target).float().mean()),
                    "target_top1_rate_after": float(
                        (modified_logits.argmax(dim=1) == target).float().mean()
                    ),
                    "matched_control_target_delta": control_delta,
                }
            )

        ablation_mask = (test_labels == target) & (
            test_latents[:, latent_id] > activation_threshold
        )
        ablation_indices = torch.where(ablation_mask)[0]
        if ablation_indices.numel() == 0:
            ablation_delta = 0.0
            ablation_off_target = 0.0
        else:
            coefficient_delta = -test_latents[ablation_indices, latent_id]
            ablated = test_reconstruction[ablation_indices] + coefficient_delta[:, None] * decoder_directions[
                latent_id
            ]
            ablated_logits = clip_zero_shot_logits(
                ablated,
                activation_mean,
                activation_std,
                text_prototypes,
                logit_scale,
            )
            delta = ablated_logits - baseline_logits[ablation_indices]
            ablation_delta = float(delta[:, target].mean())
            ablation_off_target = _off_target_absolute_change(delta, target)
        rows.append(
            {
                "latent_id": int(latent_id),
                "target_fine_class": target,
                "ms": float(ms_scores[latent_id]),
                "train_support": int(train_support[latent_id]),
                "test_support": int(test_support[latent_id]),
                "positive_quantile_clamp": clamp,
                "matched_control_latent": control_id,
                "injection_sample_count": int(injection_indices.numel()),
                "dose_response": dose_rows,
                "ablation_sample_count": int(ablation_indices.numel()),
                "ablation_mean_target_logit_delta": ablation_delta,
                "ablation_mean_off_target_absolute_delta": ablation_off_target,
            }
        )

    reference_dose = min(
        range(len(dose_values)),
        key=lambda index: abs(float(dose_values[index]) - 1.0),
    )
    positive_injections = sum(
        float(row["dose_response"][reference_dose]["mean_target_logit_delta"]) > 0 for row in rows
    )
    negative_ablations = sum(float(row["ablation_mean_target_logit_delta"]) < 0 for row in rows)
    monotonic_injections = sum(
        all(
            float(right["mean_target_logit_delta"])
            >= float(left["mean_target_logit_delta"]) - 1e-8
            for left, right in zip(row["dose_response"], row["dose_response"][1:])
        )
        and float(row["dose_response"][-1]["mean_target_logit_delta"]) > 0
        for row in rows
    )
    beats_control = sum(
        float(row["dose_response"][reference_dose]["mean_target_logit_delta"])
        > float(row["dose_response"][reference_dose]["matched_control_target_delta"])
        for row in rows
    )
    bidirectional_specific = sum(
        float(row["dose_response"][reference_dose]["mean_target_logit_delta"]) > 0
        and float(row["ablation_mean_target_logit_delta"]) < 0
        and float(row["dose_response"][reference_dose]["mean_target_logit_delta"])
        > float(row["dose_response"][reference_dose]["matched_control_target_delta"])
        for row in rows
    )
    summary = {
        "eligible_latents": int(valid.numel()),
        "evaluated_latents": len(rows),
        "reference_dose": float(dose_values[reference_dose]),
        "positive_injection_effects": int(positive_injections),
        "negative_ablation_effects": int(negative_ablations),
        "monotonic_injection_effects": int(monotonic_injections),
        "injection_beats_matched_control": int(beats_control),
        "bidirectional_specific_effects": int(bidirectional_specific),
        "causal_boundary": (
            "干预改变了 SAE 重建后的 CLIP 图文 logits；尚未注入 CLIP 中间 token，"
            "也不等同于 LLaVA 文本生成干预"
        ),
    }
    return rows, summary
