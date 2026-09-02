from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _finite_mean(values: list[float]) -> float:
    """计算有限数值的均值；没有可用值时返回 NaN。"""
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _pearson_columns(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """逐列计算 Pearson 相关，常数列记为 NaN。"""
    first_centered = first.float() - first.float().mean(dim=0, keepdim=True)
    second_centered = second.float() - second.float().mean(dim=0, keepdim=True)
    numerator = torch.sum(first_centered * second_centered, dim=0)
    denominator = torch.sqrt(
        torch.sum(first_centered**2, dim=0) * torch.sum(second_centered**2, dim=0)
    )
    result = numerator / denominator.clamp_min(1e-12)
    result[denominator <= 1e-12] = float("nan")
    return result


def _jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    """计算两个布尔集合的 Jaccard；空并集记为 NaN。"""
    union = torch.sum(first | second)
    if int(union) == 0:
        return float("nan")
    return float((torch.sum(first & second).float() / union).cpu())


def _top_mask(values: torch.Tensor, top_k: int, threshold: float) -> torch.Tensor:
    """把真正正激活的 top-k 样本转换为布尔掩码。"""
    mask = torch.zeros(len(values), dtype=torch.bool)
    positive = torch.where(values > threshold)[0]
    if positive.numel() == 0:
        return mask
    count = min(int(top_k), int(positive.numel()))
    selected = positive[torch.topk(values[positive], count).indices]
    mask[selected] = True
    return mask


def _label_profile(
    firing: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
) -> tuple[torch.Tensor, int, float]:
    """返回激活样本的标签分布、主标签和纯度。"""
    selected = labels[firing]
    if selected.numel() == 0:
        return torch.zeros(class_count), -1, float("nan")
    counts = torch.bincount(selected.long(), minlength=class_count).float()
    profile = counts / counts.sum().clamp_min(1.0)
    best = int(torch.argmax(counts))
    return profile, best, float(profile[best])


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    """计算两个标签分布的余弦相似度。"""
    denominator = torch.linalg.norm(first) * torch.linalg.norm(second)
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(first, second) / denominator)


def latent_alignment_report(
    first: torch.Tensor,
    second: torch.Tensor,
    fine_labels: torch.Tensor,
    coarse_labels: torch.Tensor,
    activation_threshold: float,
    top_k: int,
    minimum_support: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """评价两个模型同索引 latent 的样本和标签语义对齐。"""
    if first.shape != second.shape:
        raise ValueError("两个模型的 latent 形状必须完全一致")
    if len(first) != len(fine_labels) or len(first) != len(coarse_labels):
        raise ValueError("latent 与标签样本数不一致")
    fine_count = int(torch.max(fine_labels)) + 1
    coarse_count = int(torch.max(coarse_labels)) + 1
    correlations = _pearson_columns(first, second)
    rows: list[dict[str, Any]] = []
    for latent_id in range(first.shape[1]):
        fires_first = first[:, latent_id] > activation_threshold
        fires_second = second[:, latent_id] > activation_threshold
        support_first = int(fires_first.sum())
        support_second = int(fires_second.sum())
        top_first = _top_mask(first[:, latent_id], top_k, activation_threshold)
        top_second = _top_mask(second[:, latent_id], top_k, activation_threshold)
        fine_first, fine_id_first, fine_purity_first = _label_profile(
            fires_first, fine_labels, fine_count
        )
        fine_second, fine_id_second, fine_purity_second = _label_profile(
            fires_second, fine_labels, fine_count
        )
        coarse_first, coarse_id_first, coarse_purity_first = _label_profile(
            fires_first, coarse_labels, coarse_count
        )
        coarse_second, coarse_id_second, coarse_purity_second = _label_profile(
            fires_second, coarse_labels, coarse_count
        )
        supported = support_first >= minimum_support and support_second >= minimum_support
        rows.append(
            {
                "latent_id": latent_id,
                "support_first": support_first,
                "support_second": support_second,
                "supported_in_both": supported,
                "activation_pearson": float(correlations[latent_id]),
                "firing_jaccard": _jaccard(fires_first, fires_second),
                "top_image_jaccard": _jaccard(top_first, top_second),
                "fine_label_first": fine_id_first,
                "fine_label_second": fine_id_second,
                "fine_label_agreement": bool(
                    supported and fine_id_first >= 0 and fine_id_first == fine_id_second
                ),
                "fine_purity_first": fine_purity_first,
                "fine_purity_second": fine_purity_second,
                "fine_profile_cosine": _cosine(fine_first, fine_second),
                "coarse_label_first": coarse_id_first,
                "coarse_label_second": coarse_id_second,
                "coarse_label_agreement": bool(
                    supported and coarse_id_first >= 0 and coarse_id_first == coarse_id_second
                ),
                "coarse_purity_first": coarse_purity_first,
                "coarse_purity_second": coarse_purity_second,
                "coarse_profile_cosine": _cosine(coarse_first, coarse_second),
            }
        )

    supported_rows = [row for row in rows if row["supported_in_both"]]
    denominator = max(len(supported_rows), 1)
    summary = {
        "latent_count": int(first.shape[1]),
        "minimum_support": int(minimum_support),
        "supported_latents": len(supported_rows),
        "supported_latent_coverage": len(supported_rows) / max(first.shape[1], 1),
        "mean_activation_pearson": _finite_mean(
            [float(row["activation_pearson"]) for row in supported_rows]
        ),
        "mean_firing_jaccard": _finite_mean(
            [float(row["firing_jaccard"]) for row in supported_rows]
        ),
        "mean_top_image_jaccard": _finite_mean(
            [float(row["top_image_jaccard"]) for row in supported_rows]
        ),
        "mean_fine_profile_cosine": _finite_mean(
            [float(row["fine_profile_cosine"]) for row in supported_rows]
        ),
        "mean_coarse_profile_cosine": _finite_mean(
            [float(row["coarse_profile_cosine"]) for row in supported_rows]
        ),
        "fine_label_agreement_rate": sum(
            bool(row["fine_label_agreement"]) for row in supported_rows
        )
        / denominator,
        "coarse_label_agreement_rate": sum(
            bool(row["coarse_label_agreement"]) for row in supported_rows
        )
        / denominator,
    }
    return rows, summary


def _null_result(observed: float, null_values: list[float]) -> dict[str, float]:
    """汇总零假设分布，并给出单侧经验 p 值。"""
    finite = np.asarray([value for value in null_values if math.isfinite(value)], dtype=float)
    if not math.isfinite(observed) or finite.size == 0:
        return {
            "observed": float(observed),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "empirical_p_value": float("nan"),
        }
    return {
        "observed": float(observed),
        "null_mean": float(finite.mean()),
        "null_std": float(finite.std()),
        "empirical_p_value": float((1 + np.sum(finite >= observed)) / (len(finite) + 1)),
    }


def alignment_null_controls(
    first: torch.Tensor,
    second: torch.Tensor,
    fine_labels: torch.Tensor,
    coarse_labels: torch.Tensor,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    activation_threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """用样本置换和 latent 索引置换建立跨模型对齐零假设。"""
    if repeats <= 0:
        return {"repeats": 0, "sample_permutation": {}, "latent_permutation": {}}
    support_first = torch.tensor(
        [int(row["support_first"]) >= int(summary["minimum_support"]) for row in rows]
    )
    support_second = torch.tensor(
        [int(row["support_second"]) >= int(summary["minimum_support"]) for row in rows]
    )
    supported = support_first & support_second
    fires_first = first > activation_threshold
    fires_second = second > activation_threshold
    generator = torch.Generator().manual_seed(seed)
    sample_pearson: list[float] = []
    sample_jaccard: list[float] = []
    latent_fine_cosine: list[float] = []
    latent_coarse_cosine: list[float] = []
    latent_fine_agreement: list[float] = []
    latent_coarse_agreement: list[float] = []

    fine_first = torch.tensor([int(row["fine_label_first"]) for row in rows])
    fine_second = torch.tensor([int(row["fine_label_second"]) for row in rows])
    coarse_first = torch.tensor([int(row["coarse_label_first"]) for row in rows])
    coarse_second = torch.tensor([int(row["coarse_label_second"]) for row in rows])

    # 标签分布余弦在任意跨列组合上需要重新计算，因此先缓存每列的分布。
    def profiles(latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_count = int(torch.max(labels)) + 1
        result = torch.zeros((latents.shape[1], class_count), dtype=torch.float32)
        firing = latents > activation_threshold
        for latent_id in range(latents.shape[1]):
            result[latent_id] = _label_profile(
                firing[:, latent_id], labels, class_count
            )[0]
        return result

    fine_profiles_first = profiles(first, fine_labels)
    fine_profiles_second = profiles(second, fine_labels)
    coarse_profiles_first = profiles(first, coarse_labels)
    coarse_profiles_second = profiles(second, coarse_labels)
    for _ in range(repeats):
        sample_order = torch.randperm(len(second), generator=generator)
        permuted_second = second[sample_order]
        correlations = _pearson_columns(first, permuted_second)
        valid_correlation = correlations[supported & torch.isfinite(correlations)]
        sample_pearson.append(
            float(valid_correlation.mean()) if valid_correlation.numel() else float("nan")
        )
        permuted_fires = fires_second[sample_order]
        intersection = torch.sum(fires_first & permuted_fires, dim=0).float()
        union = torch.sum(fires_first | permuted_fires, dim=0).float()
        jaccard = intersection / union.clamp_min(1.0)
        valid_jaccard = jaccard[supported & (union > 0)]
        sample_jaccard.append(
            float(valid_jaccard.mean()) if valid_jaccard.numel() else float("nan")
        )

        latent_order = torch.randperm(first.shape[1], generator=generator)
        paired_support = support_first & support_second[latent_order]
        if bool(paired_support.any()):
            def paired_profile_cosine(
                profiles_first: torch.Tensor,
                profiles_second: torch.Tensor,
            ) -> float:
                selected_first = profiles_first[paired_support]
                selected_second = profiles_second[latent_order][paired_support]
                numerator = torch.sum(selected_first * selected_second, dim=1)
                denominator = torch.linalg.norm(selected_first, dim=1) * torch.linalg.norm(
                    selected_second, dim=1
                )
                valid = denominator > 1e-12
                values = numerator[valid] / denominator[valid]
                return float(values.mean()) if values.numel() else float("nan")

            latent_fine_cosine.append(
                paired_profile_cosine(fine_profiles_first, fine_profiles_second)
            )
            latent_coarse_cosine.append(
                paired_profile_cosine(coarse_profiles_first, coarse_profiles_second)
            )
            latent_fine_agreement.append(
                float(
                    (fine_first[paired_support] == fine_second[latent_order][paired_support])
                    .float()
                    .mean()
                )
            )
            latent_coarse_agreement.append(
                float(
                    (coarse_first[paired_support] == coarse_second[latent_order][paired_support])
                    .float()
                    .mean()
                )
            )
        else:
            latent_fine_cosine.append(float("nan"))
            latent_coarse_cosine.append(float("nan"))
            latent_fine_agreement.append(float("nan"))
            latent_coarse_agreement.append(float("nan"))

    return {
        "repeats": int(repeats),
        "sample_permutation": {
            "mean_activation_pearson": _null_result(
                float(summary["mean_activation_pearson"]), sample_pearson
            ),
            "mean_firing_jaccard": _null_result(
                float(summary["mean_firing_jaccard"]), sample_jaccard
            ),
        },
        "latent_permutation": {
            "mean_fine_profile_cosine": _null_result(
                float(summary["mean_fine_profile_cosine"]), latent_fine_cosine
            ),
            "mean_coarse_profile_cosine": _null_result(
                float(summary["mean_coarse_profile_cosine"]), latent_coarse_cosine
            ),
            "fine_label_agreement_rate": _null_result(
                float(summary["fine_label_agreement_rate"]), latent_fine_agreement
            ),
            "coarse_label_agreement_rate": _null_result(
                float(summary["coarse_label_agreement_rate"]), latent_coarse_agreement
            ),
        },
    }


def bootstrap_alignment_intervals(
    rows: list[dict[str, Any]],
    repeats: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """以 latent 为抽样单位估计主要对齐指标的 bootstrap 区间。"""
    supported = [row for row in rows if row["supported_in_both"]]
    metric_names = [
        "activation_pearson",
        "firing_jaccard",
        "top_image_jaccard",
        "fine_profile_cosine",
        "coarse_profile_cosine",
    ]
    if not supported or repeats <= 0:
        return {}
    generator = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray(
            [float(row[name]) for row in supported if math.isfinite(float(row[name]))],
            dtype=float,
        )
        if values.size == 0:
            continue
        boot = np.asarray(
            [generator.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)]
        )
        output[name] = {
            "mean": float(values.mean()),
            "lower": float(np.quantile(boot, alpha)),
            "upper": float(np.quantile(boot, 1.0 - alpha)),
            "confidence": float(confidence),
        }
    return output
