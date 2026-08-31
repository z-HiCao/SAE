from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .models import BatchTopKSAE, UniversalSAE


@dataclass(frozen=True)
class TrainingSpec:
    """单模型 SAE 训练所需的最小参数集合。"""

    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0


def _infinite_loader(tensor: torch.Tensor, batch_size: int) -> Any:
    """持续产生随机打乱的小批量。"""
    while True:
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
        for (batch,) in loader:
            yield batch


def train_sae(
    model: torch.nn.Module,
    train_tensor: torch.Tensor,
    spec: TrainingSpec,
    device: torch.device,
) -> list[dict[str, float]]:
    """训练单模型 SAE，并返回稀疏的日志快照。"""
    model.to(device)
    if hasattr(model, "center"):
        with torch.no_grad():
            model.center.copy_(train_tensor.mean(dim=0).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    loader = _infinite_loader(train_tensor, spec.batch_size)
    history: list[dict[str, float]] = []
    for step in trange(spec.steps, desc="训练 SAE"):
        batch = next(loader).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        if hasattr(model, "normalize_decoder_directions"):
            model.normalize_decoder_directions()
        if step % 100 == 0 or step == spec.steps - 1:
            row = {"step": float(step), "loss": float(loss.detach().cpu())}
            row.update({key: float(value.cpu()) for key, value in parts.items()})
            history.append(row)

    if isinstance(model, BatchTopKSAE):
        calibration_batches = []
        for start in range(0, min(len(train_tensor), spec.batch_size * 20), spec.batch_size):
            calibration_batches.append(train_tensor[start : start + spec.batch_size].to(device))
        model.eval()
        model.calibrate_thresholds(calibration_batches)
    return history


@torch.no_grad()
def encode_and_reconstruct(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分批编码与重建，最终统一返回 CPU tensor。"""
    model.eval().to(device)
    latents = []
    reconstructions = []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start : start + batch_size].to(device)
        z, reconstruction = model(batch)
        latents.append(z.cpu())
        reconstructions.append(reconstruction.cpu())
    return torch.cat(latents), torch.cat(reconstructions)


def train_universal_sae(
    model: UniversalSAE,
    tensors: list[torch.Tensor],
    steps: int,
    batch_size: int,
    learning_rate: float,
    loss_name: str,
    device: torch.device,
) -> list[dict[str, float]]:
    """交替选择源模型并通过两个 decoder 重建对齐激活。"""
    if len(tensors[0]) != len(tensors[1]):
        raise ValueError("两个模型的训练激活样本数不一致")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(tensors[0], tensors[1])
    history: list[dict[str, float]] = []
    iterator = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    for step in trange(steps, desc="训练 Universal SAE"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
            batch = next(iterator)
        aligned = [item.to(device) for item in batch]
        source_model = step % 2
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(source_model, aligned, loss_name)
        loss.backward()
        optimizer.step()
        model.normalize_decoder_directions()
        if step % 100 == 0 or step == steps - 1:
            history.append(
                {"step": float(step), "source_model": float(source_model), "loss": float(loss.detach().cpu())}
            )
    return history
