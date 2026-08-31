from __future__ import annotations

import torch


def r2_score(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """计算相对于均值基线的整体 R²。"""
    residual = torch.sum((original - reconstruction) ** 2)
    centered = original - original.mean(dim=0, keepdim=True)
    total = torch.sum(centered**2)
    if total <= 0:
        return float("nan")
    return float((1.0 - residual / total).cpu())


def l0_score(latents: torch.Tensor, threshold: float = 0.0) -> float:
    """计算每个样本的平均非零 latent 数。"""
    return float((latents > threshold).sum(dim=1).float().mean().cpu())


def dead_latent_fraction(latents: torch.Tensor, threshold: float = 0.0) -> float:
    """计算在整个评价集上从未激活的 latent 比例。"""
    dead = (latents > threshold).sum(dim=0) == 0
    return float(dead.float().mean().cpu())

