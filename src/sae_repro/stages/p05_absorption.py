from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import (
    load_json,
    require_file,
    save_array,
    save_json,
    write_manifest,
)
from sae_repro.core.paths import resolve_project_path
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.absorption import (
    discover_absorption_features,
    evaluate_absorption_features,
    stratified_fit_validation_indices,
)
from sae_repro.visualization.absorption import save_absorption_plots

from .common import shared_array, stage_dir, tensor


def _analyze_source(
    config: dict[str, Any],
    source_dir: Any,
    train_activations_name: str,
    train_latents_name: str,
    test_activations_name: str,
    test_latents_name: str,
    decoder_name: str,
) -> dict[str, Any]:
    """用训练发现集选择规则，并分别在验证集和独立测试集评价。"""
    train_activations = tensor(
        np.load(require_file(source_dir / train_activations_name, f"运行 {source_dir.name}"))
    )
    train_latents = tensor(
        np.load(require_file(source_dir / train_latents_name, f"运行 {source_dir.name}"))
    )
    test_activations = tensor(
        np.load(require_file(source_dir / test_activations_name, f"运行 {source_dir.name}"))
    )
    test_latents = tensor(
        np.load(require_file(source_dir / test_latents_name, f"运行 {source_dir.name}"))
    )
    decoder = tensor(np.load(require_file(source_dir / decoder_name, f"运行 {source_dir.name}")))
    train_coarse = torch.from_numpy(shared_array(config, "train", "coarse")).long()
    train_fine = torch.from_numpy(shared_array(config, "train", "fine")).long()
    test_coarse = torch.from_numpy(shared_array(config, "test", "coarse")).long()
    test_fine = torch.from_numpy(shared_array(config, "test", "fine")).long()
    if len(train_activations) != len(train_latents) or len(train_latents) != len(train_coarse):
        raise RuntimeError(f"{source_dir.name} 的训练激活、latent 和共享标签样本数不一致")
    if len(test_activations) != len(test_latents) or len(test_latents) != len(test_coarse):
        raise RuntimeError(f"{source_dir.name} 的测试激活、latent 和共享标签样本数不一致")

    fit_indices, validation_indices = stratified_fit_validation_indices(
        train_coarse,
        float(config["discovery_fit_fraction"]),
        int(config["seed"]),
    )
    discoveries = discover_absorption_features(
        train_activations[fit_indices],
        train_latents[fit_indices],
        decoder,
        train_coarse[fit_indices],
        train_fine[fit_indices],
        config,
    )
    validation = evaluate_absorption_features(
        discoveries,
        train_latents[fit_indices],
        decoder,
        train_activations[validation_indices],
        train_latents[validation_indices],
        train_coarse[validation_indices],
        train_fine[validation_indices],
        config,
        "validation",
    )
    test = evaluate_absorption_features(
        discoveries,
        train_latents[fit_indices],
        decoder,
        test_activations,
        test_latents,
        test_coarse,
        test_fine,
        config,
        "test",
    )
    discovery_rows = [item.discovery_report for item in discoveries]
    return {
        "split_protocol": {
            "discovery_samples": int(len(fit_indices)),
            "validation_samples": int(len(validation_indices)),
            "test_samples": int(len(test_latents)),
            "rule": "main latent、split 顺序、parent direction 与候选只由 discovery 确定",
        },
        "discovery": discovery_rows,
        "validation": validation,
        "test": test,
        "summary": _summary(test),
    }


def _summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总以 coarse parent 为单位的测试结果。"""
    if not reports:
        return {}
    return {
        "parent_count": len(reports),
        "parents_with_candidates": sum(int(row["candidate_count"] > 0) for row in reports),
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in reports])),
        "mean_additional_splitting_count": float(
            np.mean([row["splitting"]["additional_splitting_count"] for row in reports])
        ),
        "mean_absorption_rate_over_tested_false_negatives": float(
            np.mean([row["absorption_rate_over_tested_false_negatives"] for row in reports])
        ),
        "mean_absorption_rate_over_parent_positives": float(
            np.mean([row["absorption_rate_over_parent_positives"] for row in reports])
        ),
        "mean_matching_child_rate_over_absorbed": float(
            np.mean([row["matching_child_rate_over_absorbed"] for row in reports])
        ),
        "parents_beating_matched_null_at_0_05": sum(
            float(row["matched_random_control"]["empirical_p_value"]) <= 0.05
            for row in reports
            if np.isfinite(float(row["matched_random_control"]["empirical_p_value"]))
        ),
    }


def _attach_names(
    payload: dict[str, Any],
    coarse_names: list[str],
    fine_names: list[str],
) -> None:
    """为数值 ID 附加 CIFAR-100 可读名称。"""
    for discovery in payload["discovery"]:
        discovery["parent_name"] = coarse_names[int(discovery["parent_id"])]
        for candidate in discovery["screened_candidates"]:
            child_id = int(candidate["child_id"])
            candidate["child_name"] = fine_names[child_id] if child_id >= 0 else ""
    for split in ("validation", "test"):
        for row in payload[split]:
            row["parent_name"] = coarse_names[int(row["parent_id"])]


def _save_candidate_csv(path: Path, sources: dict[str, dict[str, Any]]) -> None:
    """把所有被筛查候选保存为便于人工审查的 CSV。"""
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for discovery in payload["discovery"]:
            for candidate in discovery["screened_candidates"]:
                rows.append(
                    {
                        "source": source,
                        "parent_id": discovery["parent_id"],
                        "parent_name": discovery["parent_name"],
                        **candidate,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "source",
        "parent_id",
        "parent_name",
        "latent_id",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any]) -> None:
    """运行第五阶段，比较受控 SAE 和双模型 SAE 中的 absorption。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    p02 = stage_dir(config, "p02")
    p04 = stage_dir(config, "p04")
    require_file(p02 / "manifest.json", "make p02")
    require_file(p04 / "manifest.json", "make p04")
    output = stage_dir(config, "p05")
    shared_train_coarse = torch.from_numpy(shared_array(config, "train", "coarse")).long()
    discovery_indices, validation_indices = stratified_fit_validation_indices(
        shared_train_coarse,
        float(config["discovery_fit_fraction"]),
        int(config["seed"]),
    )
    save_array(output / "discovery_indices.npy", discovery_indices.numpy())
    save_array(output / "validation_indices.npy", validation_indices.numpy())
    controlled = _analyze_source(
        config,
        p02,
        "train_activations.npy",
        "train_latents.npy",
        "test_activations.npy",
        "test_latents.npy",
        "decoder_directions.npy",
    )
    universal_clip = _analyze_source(
        config,
        p04,
        "train_clip_activations.npy",
        "train_latents_clip.npy",
        "test_clip_activations.npy",
        "test_latents_clip.npy",
        "decoder_clip.npy",
    )
    shared_manifest = load_json(
        require_file(
            resolve_project_path(config["paths"]["shared_root"]) / "manifest.json",
            "make prepare",
        )
    )
    coarse_names = [str(value) for value in shared_manifest["coarse_names"]]
    fine_names = [str(value) for value in shared_manifest["fine_names"]]
    _attach_names(controlled, coarse_names, fine_names)
    _attach_names(universal_clip, coarse_names, fine_names)
    payload = {
        "status": "ADAPTED",
        "controlled_p02": controlled,
        "universal_clip_p04": universal_clip,
        "mean_absorption_rate_controlled": controlled["summary"][
            "mean_absorption_rate_over_parent_positives"
        ],
        "mean_absorption_rate_universal_clip": universal_clip["summary"][
            "mean_absorption_rate_over_parent_positives"
        ],
        "interpretation_boundary": (
            "候选在 discovery 中固定，并在独立 test 上执行 decoder 重建消融；"
            "该实验支持 SAE 表示内部对线性 parent probe 的因果贡献，"
            "但不是 VLM 最终生成行为上的完整因果证明"
        ),
    }
    source_payloads = {
        "controlled_p02": controlled,
        "universal_clip_p04": universal_clip,
    }
    _save_candidate_csv(output / "candidate_latents.csv", source_payloads)
    save_absorption_plots(
        {name: value["test"] for name, value in source_payloads.items()},
        output / "plots",
    )
    save_json(output / "controlled_p02.json", controlled)
    save_json(output / "universal_clip_p04.json", universal_clip)
    save_json(output / "metrics.json", payload)
    write_manifest(
        output / "manifest.json",
        "p05",
        config,
        inputs=[str(p02 / "manifest.json"), str(p04 / "manifest.json")],
    )
