from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import require_file
from sae_repro.core.paths import resolve_project_path


def shared_array(config: dict[str, Any], split: str, name: str) -> np.ndarray:
    """读取共享数据数组，并在缺失时提示先运行 prepare。"""
    root = resolve_project_path(config["paths"]["shared_root"])
    path = require_file(root / f"{split}_{name}.npy", "make prepare")
    return np.load(path)


def standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用训练集统计量标准化训练集和测试集。"""
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def tensor(array: np.ndarray) -> torch.Tensor:
    """统一把 NumPy 数组转换为 CPU float32 tensor。"""
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def stage_dir(config: dict[str, Any], stage: str) -> Path:
    """返回某阶段产物目录。"""
    root = resolve_project_path(config["paths"]["output_root"])
    path = root / stage
    path.mkdir(parents=True, exist_ok=True)
    return path

