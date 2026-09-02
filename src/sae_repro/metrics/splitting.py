from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .concepts import binary_f1


def evaluate_splitting_order(
    firing: torch.Tensor,
    target: torch.Tensor,
    latent_order: list[int],
    f1_jump: float,
) -> dict[str, Any]:
    """在固定 latent 顺序上评价并集 F1，首个 latent 只作为基线。"""
    combined = torch.zeros_like(target, dtype=torch.bool)
    curve: list[float] = []
    increments: list[float] = []
    previous = 0.0
    for latent_id in latent_order:
        combined |= firing[:, latent_id]
        current = binary_f1(combined, target)
        curve.append(current)
        increments.append(current - previous)
        previous = current
    additional = sum(increment > f1_jump for increment in increments[1:])
    return {
        "latent_order": [int(value) for value in latent_order],
        "union_f1_curve": curve,
        "f1_increments": increments,
        "baseline_f1": curve[0] if curve else 0.0,
        "additional_splitting_count": int(additional),
        "definition": "首个 latent 是基线，只有后续并集 F1 的显著增益才计为 splitting",
    }


def discover_splitting_order(
    firing: torch.Tensor,
    target: torch.Tensor,
    max_latents: int,
    f1_jump: float,
) -> dict[str, Any]:
    """仅在发现集按单 latent F1 排序，然后计算 splitting 曲线。"""
    individual = [binary_f1(firing[:, idx], target) for idx in range(firing.shape[1])]
    # 稳定降序保证并列时与 np.argmax 选择同一个最小索引主 latent。
    order = np.argsort(-np.asarray(individual), kind="stable")[:max_latents].tolist()
    report = evaluate_splitting_order(firing, target, order, f1_jump)
    report["individual_f1_in_order"] = [float(individual[index]) for index in order]
    return report
