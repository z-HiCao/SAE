from __future__ import annotations

import math

import torch

from .reconstruction import r2_score


@torch.no_grad()
def cross_reconstruction_matrix(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """计算从模型 i 编码、由模型 j 解码的 2×2 R² 矩阵。"""
    model.eval().to(device)
    matrix = torch.zeros((2, 2), dtype=torch.float32)
    for source in range(2):
        codes = []
        for start in range(0, len(inputs[source]), batch_size):
            codes.append(model.encode(source, inputs[source][start : start + batch_size].to(device)).cpu())
        z = torch.cat(codes)
        for target in range(2):
            reconstructions = []
            for start in range(0, len(z), batch_size):
                reconstructions.append(model.decode(target, z[start : start + batch_size].to(device)).cpu())
            matrix[source, target] = r2_score(inputs[target], torch.cat(reconstructions))
    return matrix


@torch.no_grad()
def shuffled_target_cross_reconstruction_matrix(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    permutation: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """以打乱后的目标样本计算交叉重构 R²，作为配对关系负对照。"""
    model.eval().to(device)
    matrix = torch.zeros((2, 2), dtype=torch.float32)
    for source in range(2):
        codes = []
        for start in range(0, len(inputs[source]), batch_size):
            batch = inputs[source][start : start + batch_size].to(device)
            codes.append(model.encode(source, batch).cpu())
        z = torch.cat(codes)
        for target in range(2):
            reconstructions = []
            for start in range(0, len(z), batch_size):
                reconstructions.append(
                    model.decode(target, z[start : start + batch_size].to(device)).cpu()
                )
            shuffled_target = inputs[target][permutation]
            matrix[source, target] = r2_score(shuffled_target, torch.cat(reconstructions))
    return matrix


def firing_entropy(latents: list[torch.Tensor], threshold: float = 0.0) -> torch.Tensor:
    """计算每个共享 latent 在两个模型间的归一化 firing entropy。"""
    counts = torch.stack([(item > threshold).sum(dim=0).float() for item in latents])
    probabilities = counts / counts.sum(dim=0, keepdim=True).clamp_min(1.0)
    safe = probabilities.clamp_min(1e-12)
    entropy = -(probabilities * torch.log(safe)).sum(dim=0) / math.log(len(latents))
    entropy[counts.sum(dim=0) == 0] = float("nan")
    return entropy


def cofire_proportion(latents: list[torch.Tensor], threshold: float = 0.0) -> torch.Tensor:
    """计算两个模型在同一样本上共同激活的比例。"""
    fires = [(item > threshold) for item in latents]
    cofire = fires[0] & fires[1]
    values = []
    for firing in fires:
        values.append(cofire.sum(dim=0).float() / firing.sum(dim=0).float().clamp_min(1.0))
    return torch.stack(values)


def concept_energy(latent: torch.Tensor, decoder_directions: torch.Tensor) -> torch.Tensor:
    """计算平均激活乘 decoder direction 后的平方 L2 能量。"""
    contribution = latent.mean(dim=0)[:, None] * decoder_directions
    return torch.sum(contribution**2, dim=1)
