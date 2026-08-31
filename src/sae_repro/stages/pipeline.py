from __future__ import annotations

from sae_repro.core.config import load_config
from sae_repro.data.cifar100 import prepare_shared_cifar100

from . import (
    p01_superposition,
    p02_interpretable_sae,
    p03_single_vlm,
    p04_universal_sae,
    p05_absorption,
)


def run_prepare() -> None:
    """只准备共享数据，不启动任何训练。"""
    prepare_shared_cifar100(load_config())


def run_stage(stage: str) -> None:
    """运行一个指定阶段。"""
    modules = {
        "p01": p01_superposition,
        "p02": p02_interpretable_sae,
        "p03": p03_single_vlm,
        "p04": p04_universal_sae,
        "p05": p05_absorption,
    }
    if stage not in modules:
        raise ValueError(f"未知阶段：{stage}")
    modules[stage].run(load_config(stage))


def run_all() -> None:
    """按严格顺序运行共享数据和五篇论文阶段。"""
    run_prepare()
    for stage in ("p01", "p02", "p03", "p04", "p05"):
        run_stage(stage)

