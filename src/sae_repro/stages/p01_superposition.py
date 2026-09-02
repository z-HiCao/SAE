from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from sae_repro.core.artifacts import save_array, save_json, write_manifest
from sae_repro.core.device import choose_device
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.data.cifar100 import prepare_shared_cifar100

from .common import shared_array, stage_dir, tensor


class ToyTiedReconstructor(nn.Module):
    """对应 Toy Models 的低维 tied-weight 重建模型。"""

    def __init__(self, input_dim: int, hidden_dim: int, relu_output: bool) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relu_output = relu_output
        self.weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """把已知概念向量压缩到低维瓶颈。"""
        return F.linear(x, self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(x)
        reconstruction = F.linear(hidden, self.weight.T, self.bias)
        if self.relu_output:
            reconstruction = F.relu(reconstruction)
        return hidden, reconstruction


def _density_variant(concepts: torch.Tensor, density: float, seed: int) -> torch.Tensor:
    """只在原本为零的概念位加入受控激活，用于稀疏度扫描。"""
    if density <= 0:
        return concepts.clone()
    generator = torch.Generator().manual_seed(seed)
    additions = torch.rand(concepts.shape, generator=generator) < density
    values = torch.rand(concepts.shape, generator=generator)
    return torch.where((concepts == 0) & additions, values, concepts)


def _train_model(
    model: ToyTiedReconstructor,
    data: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    """训练单个 toy reconstruction 设置。"""
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    iterator = iter(DataLoader(TensorDataset(data), batch_size=int(config["batch_size"]), shuffle=True))
    for step in trange(int(config["steps"]), desc="训练 superposition toy model"):
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(
                DataLoader(TensorDataset(data), batch_size=int(config["batch_size"]), shuffle=True)
            )
            (batch,) = next(iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, reconstruction = model(batch)
        loss = F.mse_loss(reconstruction, batch)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == int(config["steps"]) - 1:
            history.append({"step": float(step), "loss": float(loss.detach().cpu())})
    return history


@torch.no_grad()
def _model_metrics(
    model: ToyTiedReconstructor,
    data: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> dict[str, float | bool]:
    """计算被表示特征数、方向干扰和重建误差。"""
    model.eval()
    _, reconstruction = model(data.to(device))
    directions = model.weight.T.detach()
    norms = torch.linalg.norm(directions, dim=1)
    normalized = F.normalize(directions, dim=1)
    gram = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(gram), dtype=torch.bool, device=gram.device)
    represented = int((norms > threshold).sum().cpu())
    interference = float(gram[off_diagonal].abs().mean().cpu())
    return {
        "mse": float(F.mse_loss(reconstruction, data.to(device)).cpu()),
        "represented_features": represented,
        "hidden_dim": int(model.hidden_dim),
        "mean_absolute_interference": interference,
        "superposition_detected": bool(represented > model.hidden_dim and interference > 1e-3),
    }


def run(config: dict[str, Any]) -> None:
    """运行第一阶段，并把低维瓶颈激活交给第二阶段。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    prepare_shared_cifar100(config)
    output = stage_dir(config, "p01")
    train_concepts = tensor(shared_array(config, "train", "concepts"))
    test_concepts = tensor(shared_array(config, "test", "concepts"))
    device = choose_device(str(config["device"]))
    results = []
    selected_model: ToyTiedReconstructor | None = None

    for density_index, density in enumerate(config["noise_densities"]):
        variant = _density_variant(train_concepts, float(density), int(config["seed"]) + density_index)
        for relu_output in (False, True):
            model = ToyTiedReconstructor(
                input_dim=train_concepts.shape[1],
                hidden_dim=int(config["hidden_dim"]),
                relu_output=relu_output,
            )
            history = _train_model(model, variant, config, device)
            metrics = _model_metrics(
                model,
                variant,
                float(config["represented_norm_threshold"]),
                device,
            )
            metrics.update(
                {
                    "noise_density": float(density),
                    "model": "relu_output" if relu_output else "linear_output",
                    "final_train_loss": history[-1]["loss"],
                }
            )
            results.append(metrics)
            if float(density) == 0.0 and relu_output:
                selected_model = model.to("cpu")

    if selected_model is None:
        raise RuntimeError("没有找到原始稀疏设置下的 ReLU 模型")
    selected_model.eval()
    with torch.no_grad():
        train_hidden, _ = selected_model(train_concepts)
        test_hidden, _ = selected_model(test_concepts)
        feature_directions = selected_model.weight.T.detach()
    save_array(output / "train_hidden.npy", train_hidden.numpy())
    save_array(output / "test_hidden.npy", test_hidden.numpy())
    save_array(output / "feature_directions.npy", feature_directions.numpy())
    torch.save(
        {
            "state_dict": selected_model.state_dict(),
            "input_dim": selected_model.input_dim,
            "hidden_dim": selected_model.hidden_dim,
            "relu_output": selected_model.relu_output,
        },
        output / "toy_model.pt",
    )
    save_json(output / "metrics.json", {"experiments": results, "status": "ADAPTED"})
    write_manifest(output / "manifest.json", "p01", config)

