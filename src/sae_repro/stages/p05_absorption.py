from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import require_file, save_json, write_manifest
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.absorption import absorption_report

from .common import shared_array, stage_dir, tensor


def _analyze_source(
    config: dict[str, Any],
    source_dir: Any,
    activations_name: str,
    latents_name: str,
    decoder_name: str,
) -> list[dict[str, Any]]:
    """读取一个上游 SAE 表示并在统一层级标签上分析 absorption。"""
    activations = tensor(
        np.load(require_file(source_dir / activations_name, f"运行 {source_dir.name}"))
    )
    latents = tensor(np.load(require_file(source_dir / latents_name, f"运行 {source_dir.name}")))
    decoder = tensor(np.load(require_file(source_dir / decoder_name, f"运行 {source_dir.name}")))
    coarse = torch.from_numpy(shared_array(config, "train", "coarse")).long()
    fine = torch.from_numpy(shared_array(config, "train", "fine")).long()
    if len(activations) != len(latents) or len(latents) != len(coarse):
        raise RuntimeError(f"{source_dir.name} 的激活、latent 和共享标签样本数不一致")
    return absorption_report(activations, latents, decoder, coarse, fine, config)


def run(config: dict[str, Any]) -> None:
    """运行第五阶段，比较受控 SAE 和双模型 SAE 中的 absorption。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    p02 = stage_dir(config, "p02")
    p04 = stage_dir(config, "p04")
    require_file(p02 / "manifest.json", "make p02")
    require_file(p04 / "manifest.json", "make p04")
    output = stage_dir(config, "p05")
    controlled = _analyze_source(
        config,
        p02,
        "train_activations.npy",
        "train_latents.npy",
        "decoder_directions.npy",
    )
    universal_clip = _analyze_source(
        config,
        p04,
        "train_clip_activations.npy",
        "train_latents_clip.npy",
        "decoder_clip.npy",
    )
    payload = {
        "status": "ADAPTED",
        "controlled_p02": controlled,
        "universal_clip_p04": universal_clip,
        "mean_absorption_rate_controlled": float(
            np.mean([row["absorption_rate_over_parent_positives"] for row in controlled])
        ),
        "mean_absorption_rate_universal_clip": float(
            np.mean([row["absorption_rate_over_parent_positives"] for row in universal_clip])
        ),
        "interpretation_boundary": (
            "该指标证明 decoder contribution 对线性 parent probe 的消融效应；"
            "它不是 VLM 最终生成行为上的完整因果证明"
        ),
    }
    save_json(output / "metrics.json", payload)
    write_manifest(
        output / "manifest.json",
        "p05",
        config,
        inputs=[str(p02 / "manifest.json"), str(p04 / "manifest.json")],
    )

