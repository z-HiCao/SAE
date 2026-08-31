from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ReLUSAE(nn.Module):
    """带 L1 稀疏惩罚的基础 ReLU SAE。"""

    def __init__(self, input_dim: int, latent_dim: int, l1_coefficient: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.l1_coefficient = l1_coefficient
        self.center = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """把原激活映射为非负稀疏系数。"""
        return F.relu(self.encoder(x - self.center))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """用 decoder dictionary 重建原激活。"""
        return self.decoder(z) + self.center

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z, reconstruction = self(x)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = z.abs().mean()
        total = reconstruction_loss + self.l1_coefficient * sparsity_loss
        return total, {
            "reconstruction_loss": reconstruction_loss.detach(),
            "sparsity_loss": sparsity_loss.detach(),
        }

    def decoder_directions(self) -> torch.Tensor:
        """返回数学记法下形状为 latent×input 的 decoder directions。"""
        return self.decoder.weight.T

    @torch.no_grad()
    def normalize_decoder_directions(self) -> None:
        """把每个 decoder direction 约束为单位范数，避免用缩放逃避 L1。"""
        norms = torch.linalg.norm(self.decoder.weight, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)


class BatchTopKSAE(nn.Module):
    """训练时在整个 batch 中保留约 B×K 个激活的 SAE。"""

    def __init__(self, input_dim: int, latent_dim: int, k: int) -> None:
        super().__init__()
        if not 0 < k <= latent_dim:
            raise ValueError("K 必须位于 1 和 latent_dim 之间")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.center = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.register_buffer("thresholds", torch.full((latent_dim,), float("nan")))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
            self.decoder.weight.copy_(self.encoder.weight.T)

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """返回稀疏化前的 latent pre-activations。"""
        return self.encoder(x - self.center)

    def _batch_topk(self, pre: torch.Tensor) -> torch.Tensor:
        """在整个 batch 展平后选择最大 B×K 项。"""
        flat = pre.flatten()
        count = min(pre.shape[0] * self.k, flat.numel())
        values, indices = torch.topk(flat, count)
        sparse = torch.zeros_like(flat)
        sparse[indices] = F.relu(values)
        return sparse.view_as(pre)

    def _sample_topk(self, pre: torch.Tensor) -> torch.Tensor:
        """没有校准阈值时提供确定性的逐样本推理后备。"""
        values, indices = torch.topk(pre, self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, indices, F.relu(values))
        return sparse

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre_activations(x)
        if self.training:
            return self._batch_topk(pre)
        if torch.isfinite(self.thresholds).all():
            return F.relu(pre - self.thresholds)
        return self._sample_topk(pre)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.center

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z, reconstruction = self(x)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        return reconstruction_loss, {"reconstruction_loss": reconstruction_loss.detach()}

    @torch.no_grad()
    def calibrate_thresholds(self, batches: list[torch.Tensor]) -> None:
        """按每批被选激活的最小正值估计每个 latent 的推理阈值。"""
        observed: list[list[torch.Tensor]] = [[] for _ in range(self.latent_dim)]
        for batch in batches:
            pre = self.pre_activations(batch)
            selected = self._batch_topk(pre)
            for latent_id in range(self.latent_dim):
                values = pre[:, latent_id][selected[:, latent_id] > 0]
                if values.numel() > 0:
                    observed[latent_id].append(values.min())
        calibrated = []
        for latent_values in observed:
            if latent_values:
                calibrated.append(torch.stack(latent_values).mean())
            else:
                calibrated.append(torch.tensor(1e6, device=self.thresholds.device))
        self.thresholds.copy_(torch.stack(calibrated))

    def decoder_directions(self) -> torch.Tensor:
        return self.decoder.weight.T

    @torch.no_grad()
    def normalize_decoder_directions(self) -> None:
        """把每个 decoder direction 约束为单位范数。"""
        norms = torch.linalg.norm(self.decoder.weight, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)


class MatryoshkaBatchTopKSAE(BatchTopKSAE):
    """在多个 latent 前缀上同时计算重建目标。"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int,
        fractions: list[float],
    ) -> None:
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, k=k)
        groups = sorted({max(1, min(latent_dim, round(latent_dim * value))) for value in fractions})
        if groups[-1] != latent_dim:
            groups.append(latent_dim)
        self.groups = groups

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(x)
        losses = []
        for group_size in self.groups:
            masked = torch.zeros_like(z)
            masked[:, :group_size] = z[:, :group_size]
            losses.append(F.mse_loss(self.decode(masked), x))
        total = torch.stack(losses).mean()
        return total, {
            "reconstruction_loss": losses[-1].detach(),
            "matryoshka_loss": total.detach(),
        }


class UniversalSAE(nn.Module):
    """两个模型专属 encoder/decoder 共享同一 latent 索引空间。"""

    def __init__(self, input_dims: list[int], latent_dim: int, k: int) -> None:
        super().__init__()
        if len(input_dims) != 2:
            raise ValueError("初始版本只实现两个模型的 Universal SAE")
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.k = k
        self.encoders = nn.ModuleList([nn.Linear(dim, latent_dim) for dim in input_dims])
        self.decoders = nn.ModuleList([nn.Linear(latent_dim, dim) for dim in input_dims])

    def encode(self, model_id: int, x: torch.Tensor) -> torch.Tensor:
        """把指定模型激活映射到共享 TopK latent。"""
        pre = F.relu(self.encoders[model_id](x))
        values, indices = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, indices, values)
        return z

    def decode(self, model_id: int, z: torch.Tensor) -> torch.Tensor:
        """用目标模型 decoder 从共享 latent 重建目标激活。"""
        return self.decoders[model_id](z)

    def compute_loss(
        self,
        source_model: int,
        inputs: list[torch.Tensor],
        loss_name: str,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """从一个模型编码，并同时重建两个模型的对齐激活。"""
        z = self.encode(source_model, inputs[source_model])
        reconstructions = [self.decode(target, z) for target in range(2)]
        if loss_name == "l1":
            losses = [F.l1_loss(reconstructions[i], inputs[i]) for i in range(2)]
        elif loss_name == "l2":
            losses = [F.mse_loss(reconstructions[i], inputs[i]) for i in range(2)]
        else:
            raise ValueError(f"未知重建损失：{loss_name}")
        return torch.stack(losses).sum(), reconstructions

    def decoder_directions(self, model_id: int) -> torch.Tensor:
        """返回指定模型形状为 latent×input 的 decoder directions。"""
        return self.decoders[model_id].weight.T

    @torch.no_grad()
    def normalize_decoder_directions(self) -> None:
        """分别归一化两个模型的 decoder directions。"""
        for decoder in self.decoders:
            norms = torch.linalg.norm(decoder.weight, dim=0, keepdim=True).clamp_min(1e-8)
            decoder.weight.div_(norms)
