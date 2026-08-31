from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from sae_repro.core.artifacts import save_array, save_json
from sae_repro.core.paths import resolve_project_path

from .concepts import build_concept_matrix, coarse_names, fine_to_coarse_map


@dataclass(frozen=True)
class SharedDatasetPaths:
    """统一列出五阶段共享的数据文件。"""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def array(self, split: str, name: str) -> Path:
        return self.root / f"{split}_{name}.npy"


def _select_indices(total: int, limit: int | None, seed: int) -> np.ndarray:
    """用固定随机种子选择并排序样本索引。"""
    if limit is None or limit <= 0 or limit >= total:
        return np.arange(total, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(total, size=limit, replace=False)).astype(np.int64)


def prepare_shared_cifar100(config: dict[str, Any]) -> SharedDatasetPaths:
    """下载 CIFAR-100 并生成跨阶段共享的索引、标签和概念矩阵。"""
    data_root = resolve_project_path(config["paths"]["data_root"])
    shared_root = resolve_project_path(config["paths"]["shared_root"])
    raw_root = data_root / "raw"
    shared_root.mkdir(parents=True, exist_ok=True)
    paths = SharedDatasetPaths(shared_root)

    train_set = CIFAR100(root=raw_root, train=True, download=config["dataset"]["download"])
    test_set = CIFAR100(root=raw_root, train=False, download=config["dataset"]["download"])
    if train_set.classes != test_set.classes:
        raise RuntimeError("CIFAR-100 训练集和测试集类别顺序不一致")

    mapping = fine_to_coarse_map(train_set.classes)
    train_indices = _select_indices(
        len(train_set), config["dataset"].get("train_limit"), int(config["seed"])
    )
    test_indices = _select_indices(
        len(test_set), config["dataset"].get("test_limit"), int(config["seed"]) + 1
    )

    def save_split(split: str, dataset: CIFAR100, indices: np.ndarray) -> None:
        fine = np.asarray(dataset.targets, dtype=np.int64)[indices]
        coarse = mapping[fine]
        concepts = build_concept_matrix(fine, coarse)
        save_array(paths.array(split, "indices"), indices)
        save_array(paths.array(split, "fine"), fine)
        save_array(paths.array(split, "coarse"), coarse)
        save_array(paths.array(split, "concepts"), concepts)

    save_split("train", train_set, train_indices)
    save_split("test", test_set, test_indices)
    save_json(
        paths.manifest,
        {
            "dataset": "CIFAR-100",
            "seed": int(config["seed"]),
            "fine_names": list(train_set.classes),
            "coarse_names": coarse_names(),
            "concept_dim": 120,
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
            "relationship": "每张图激活一个 fine concept 和其唯一 coarse parent concept",
        },
    )
    return paths


class IndexedCIFAR100(Dataset):
    """按共享 manifest 中的索引返回 PIL 图像和层级标签。"""

    def __init__(self, config: dict[str, Any], split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"未知 split：{split}")
        data_root = resolve_project_path(config["paths"]["data_root"])
        shared_root = resolve_project_path(config["paths"]["shared_root"])
        self.base = CIFAR100(
            root=data_root / "raw",
            train=split == "train",
            download=config["dataset"]["download"],
        )
        self.indices = np.load(shared_root / f"{split}_indices.npy")
        self.fine = np.load(shared_root / f"{split}_fine.npy")
        self.coarse = np.load(shared_root / f"{split}_coarse.npy")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Any, int, int, int]:
        base_index = int(self.indices[item])
        image, fine = self.base[base_index]
        if int(fine) != int(self.fine[item]):
            raise RuntimeError("共享标签与 torchvision 样本不一致")
        return image, int(self.fine[item]), int(self.coarse[item]), base_index

