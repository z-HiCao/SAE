from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import load_json, require_file, save_array, save_json, write_manifest
from sae_repro.core.device import choose_device, release_device_cache
from sae_repro.core.paths import resolve_project_path
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.cross_model_alignment import (
    alignment_null_controls,
    bootstrap_alignment_intervals,
    latent_alignment_report,
)
from sae_repro.metrics.universality import (
    cofire_proportion,
    concept_energy,
    cross_reconstruction_matrix,
    firing_entropy,
    shuffled_target_cross_reconstruction_matrix,
)
from sae_repro.models.vision import extract_vision_embeddings
from sae_repro.sae.models import UniversalSAE
from sae_repro.sae.trainer import train_universal_sae
from sae_repro.visualization.cross_model import save_cross_model_alignment_plots

from .common import shared_array, stage_dir, standardize_train_test, tensor


@torch.no_grad()
def _encode_model(
    model: UniversalSAE,
    model_id: int,
    values: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """分批把一个模型激活编码到共享 latent。"""
    model.eval().to(device)
    rows = []
    for start in range(0, len(values), batch_size):
        rows.append(model.encode(model_id, values[start : start + batch_size].to(device)).cpu())
    return torch.cat(rows)


def _load_or_extract_siglip(
    config: dict[str, Any],
    output: Any,
    split: str,
) -> np.ndarray:
    """复用 SigLIP 缓存，否则按同一共享索引提取。"""
    path = output / f"{split}_siglip_raw.npy"
    id_path = output / f"{split}_siglip_sample_ids.npy"
    if path.exists() and id_path.exists():
        return np.load(path)
    return extract_vision_embeddings(
        config=config,
        split=split,
        model_name=str(config["second_model_name"]),
        model_kind=str(config["second_model_kind"]),
        batch_size=int(config["model_batch_size"]),
        output_path=path,
        id_path=id_path,
    )


def _train_model(
    config: dict[str, Any],
    train_inputs: list[torch.Tensor],
    device: torch.device,
    seed_offset: int = 0,
) -> tuple[UniversalSAE, list[dict[str, float]]]:
    """按统一超参数训练一个 Universal SAE，便于构造匹配对照。"""
    seed_everything(int(config["seed"]) + seed_offset)
    model = UniversalSAE(
        input_dims=[int(item.shape[1]) for item in train_inputs],
        latent_dim=int(config["latent_dim"]),
        k=int(config["k"]),
    )
    history = train_universal_sae(
        model,
        train_inputs,
        steps=int(config["steps"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        loss_name=str(config["reconstruction_loss"]),
        device=device,
    )
    return model, history


def _evaluate_alignment(
    config: dict[str, Any],
    latents: list[torch.Tensor],
    fine_labels: torch.Tensor,
    coarse_labels: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """使用统一协议评价两个模型的同索引 latent。"""
    return latent_alignment_report(
        latents[0],
        latents[1],
        fine_labels,
        coarse_labels,
        activation_threshold=float(config["activation_threshold"]),
        top_k=int(config["alignment_top_k"]),
        minimum_support=int(config["minimum_alignment_support"]),
    )


def _save_alignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存逐 latent 对齐表，便于排序和人工审查。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _attach_label_names(
    rows: list[dict[str, Any]],
    fine_names: list[str],
    coarse_names: list[str],
) -> None:
    """给逐 latent 表中的主标签 ID 补充可读名称。"""
    for row in rows:
        for side in ("first", "second"):
            fine_id = int(row[f"fine_label_{side}"])
            coarse_id = int(row[f"coarse_label_{side}"])
            row[f"fine_label_name_{side}"] = fine_names[fine_id] if fine_id >= 0 else ""
            row[f"coarse_label_name_{side}"] = (
                coarse_names[coarse_id] if coarse_id >= 0 else ""
            )


def run(config: dict[str, Any]) -> None:
    """运行第四阶段，学习 CLIP 与 SigLIP 的共享 latent。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    p03 = stage_dir(config, "p03")
    require_file(p03 / "manifest.json", "make p03")
    output = stage_dir(config, "p04")
    clip_train = tensor(np.load(require_file(p03 / "train_activations.npy", "make p03")))
    clip_test = tensor(np.load(require_file(p03 / "test_activations.npy", "make p03")))
    siglip_train_raw = _load_or_extract_siglip(config, output, "train")
    siglip_test_raw = _load_or_extract_siglip(config, output, "test")
    siglip_train, siglip_test, siglip_mean, siglip_std = standardize_train_test(
        siglip_train_raw, siglip_test_raw
    )
    siglip_train_tensor = tensor(siglip_train)
    siglip_test_tensor = tensor(siglip_test)

    for split in ("train", "test"):
        clip_ids = np.load(require_file(p03 / f"{split}_sample_ids.npy", "make p03"))
        siglip_ids = np.load(
            require_file(output / f"{split}_siglip_sample_ids.npy", "重新提取 SigLIP 激活")
        )
        expected_ids = shared_array(config, split, "indices")
        if not np.array_equal(clip_ids, siglip_ids) or not np.array_equal(clip_ids, expected_ids):
            raise RuntimeError(f"CLIP、SigLIP 与共享 CIFAR-100 {split} 样本顺序不一致")

    device = choose_device(str(config["device"]))
    model, history = _train_model(
        config,
        [clip_train, siglip_train_tensor],
        device,
    )
    train_latents = [
        _encode_model(model, 0, clip_train, int(config["batch_size"]), device),
        _encode_model(model, 1, siglip_train_tensor, int(config["batch_size"]), device),
    ]
    test_latents = [
        _encode_model(model, 0, clip_test, int(config["batch_size"]), device),
        _encode_model(model, 1, siglip_test_tensor, int(config["batch_size"]), device),
    ]
    cross_r2 = cross_reconstruction_matrix(
        model,
        [clip_test, siglip_test_tensor],
        device=device,
        batch_size=int(config["batch_size"]),
    )
    activation_threshold = float(config["activation_threshold"])
    entropy = firing_entropy(test_latents, threshold=activation_threshold)
    cofire = cofire_proportion(test_latents, threshold=activation_threshold)
    energy_clip = concept_energy(
        train_latents[0], model.decoder_directions(0).detach().cpu()
    )
    energy_siglip = concept_energy(
        train_latents[1], model.decoder_directions(1).detach().cpu()
    )
    fine_labels = torch.from_numpy(shared_array(config, "test", "fine")).long()
    coarse_labels = torch.from_numpy(shared_array(config, "test", "coarse")).long()
    alignment_rows, alignment_summary = _evaluate_alignment(
        config, test_latents, fine_labels, coarse_labels
    )
    shared_manifest = load_json(
        require_file(
            resolve_project_path(config["paths"]["shared_root"]) / "manifest.json",
            "make prepare",
        )
    )
    fine_names = [str(value) for value in shared_manifest["fine_names"]]
    coarse_names = [str(value) for value in shared_manifest["coarse_names"]]
    _attach_label_names(alignment_rows, fine_names, coarse_names)
    null_controls = alignment_null_controls(
        test_latents[0],
        test_latents[1],
        fine_labels,
        coarse_labels,
        alignment_rows,
        alignment_summary,
        activation_threshold=float(config["activation_threshold"]),
        repeats=int(config["alignment_null_repeats"]),
        seed=int(config["seed"]) + 101,
    )
    alignment_intervals = bootstrap_alignment_intervals(
        alignment_rows,
        repeats=int(config["bootstrap_repeats"]),
        confidence=float(config["bootstrap_confidence"]),
        seed=int(config["seed"]) + 202,
    )
    generator = torch.Generator().manual_seed(int(config["seed"]) + 303)
    target_permutation = torch.randperm(len(clip_test), generator=generator)
    save_array(output / "shuffled_target_test_permutation.npy", target_permutation.numpy())
    shuffled_target_r2 = shuffled_target_cross_reconstruction_matrix(
        model,
        [clip_test, siglip_test_tensor],
        target_permutation,
        device=device,
        batch_size=int(config["batch_size"]),
    )

    shuffled_training_control: dict[str, Any] = {"enabled": False}
    shuffled_model: UniversalSAE | None = None
    if bool(config.get("train_shuffled_pair_control", True)):
        train_permutation = torch.randperm(len(clip_train), generator=generator)
        if torch.equal(train_permutation, torch.arange(len(clip_train))):
            train_permutation = torch.roll(train_permutation, shifts=1)
        save_array(output / "shuffled_pair_train_permutation.npy", train_permutation.numpy())
        shuffled_model, shuffled_history = _train_model(
            config,
            [clip_train, siglip_train_tensor[train_permutation]],
            device,
            seed_offset=1,
        )
        shuffled_test_latents = [
            _encode_model(
                shuffled_model, model_id, values, int(config["batch_size"]), device
            )
            for model_id, values in enumerate([clip_test, siglip_test_tensor])
        ]
        shuffled_rows, shuffled_summary = _evaluate_alignment(
            config, shuffled_test_latents, fine_labels, coarse_labels
        )
        _attach_label_names(shuffled_rows, fine_names, coarse_names)
        shuffled_cross_r2 = cross_reconstruction_matrix(
            shuffled_model,
            [clip_test, siglip_test_tensor],
            device=device,
            batch_size=int(config["batch_size"]),
        )
        shuffled_training_control = {
            "enabled": True,
            "pairing": "训练时仅打乱 SigLIP 样本顺序；测试仍使用正确配对",
            "cross_reconstruction_r2": shuffled_cross_r2.tolist(),
            "alignment_summary": shuffled_summary,
            "train_history": shuffled_history,
            "comparison": {
                key: float(alignment_summary[key]) - float(shuffled_summary[key])
                for key in (
                    "mean_activation_pearson",
                    "mean_firing_jaccard",
                    "mean_top_image_jaccard",
                    "mean_fine_profile_cosine",
                    "mean_coarse_profile_cosine",
                    "fine_label_agreement_rate",
                    "coarse_label_agreement_rate",
                )
            },
        }
        save_json(output / "shuffled_pair_latent_alignment.json", shuffled_rows)

    metrics = {
        "status": "ADAPTED",
        "cross_reconstruction_r2": cross_r2.tolist(),
        "shuffled_target_cross_reconstruction_r2": shuffled_target_r2.tolist(),
        "mean_firing_entropy": float(torch.nanmean(entropy)),
        "mean_cofire_clip": float(cofire[0].mean()),
        "mean_cofire_siglip": float(cofire[1].mean()),
        "mean_energy_clip": float(energy_clip.mean()),
        "mean_energy_siglip": float(energy_siglip.mean()),
        "latent_alignment": alignment_summary,
        "latent_alignment_bootstrap_intervals": alignment_intervals,
        "alignment_null_controls": null_controls,
        "shuffled_pair_training_control": shuffled_training_control,
        "train_history": history,
        "semantic_boundary": (
            "只有正确配对训练相对样本置换、latent 置换和打乱配对训练均表现更好，"
            "且同索引 latent 在独立测试集上具有标签分布与 top 图像一致性，"
            "结果才支持跨模型语义共享；仍不能据此断言两个 VLM 内部机制完全相同"
        ),
    }
    save_array(output / "train_clip_activations.npy", clip_train.numpy())
    save_array(output / "test_clip_activations.npy", clip_test.numpy())
    save_array(output / "train_siglip_activations.npy", siglip_train)
    save_array(output / "test_siglip_activations.npy", siglip_test)
    save_array(output / "siglip_mean.npy", siglip_mean)
    save_array(output / "siglip_std.npy", siglip_std)
    save_array(output / "train_latents_clip.npy", train_latents[0].numpy())
    save_array(output / "train_latents_siglip.npy", train_latents[1].numpy())
    save_array(output / "test_latents_clip.npy", test_latents[0].numpy())
    save_array(output / "test_latents_siglip.npy", test_latents[1].numpy())
    save_array(output / "decoder_clip.npy", model.decoder_directions(0).detach().cpu().numpy())
    save_array(output / "decoder_siglip.npy", model.decoder_directions(1).detach().cpu().numpy())
    save_array(output / "firing_entropy.npy", entropy.numpy())
    save_array(output / "cofire_proportion.npy", cofire.numpy())
    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "input_dims": model.input_dims,
            "latent_dim": model.latent_dim,
            "k": model.k,
        },
        output / "universal_sae.pt",
    )
    if shuffled_model is not None and bool(config.get("save_control_checkpoint", False)):
        torch.save(
            {
                "state_dict": shuffled_model.to("cpu").state_dict(),
                "input_dims": shuffled_model.input_dims,
                "latent_dim": shuffled_model.latent_dim,
                "k": shuffled_model.k,
                "control": "shuffled_pair_training",
            },
            output / "universal_sae_shuffled_control.pt",
        )
    save_json(
        output / "latent_alignment.json",
        {"summary": alignment_summary, "rows": alignment_rows},
    )
    _save_alignment_csv(output / "latent_alignment.csv", alignment_rows)
    save_json(output / "universality_controls.json", {
        "alignment_null_controls": null_controls,
        "shuffled_target_cross_reconstruction_r2": shuffled_target_r2.tolist(),
        "shuffled_pair_training_control": shuffled_training_control,
    })
    save_cross_model_alignment_plots(
        alignment_rows,
        null_controls,
        output / "alignment_plots",
    )
    save_json(output / "metrics.json", metrics)
    write_manifest(output / "manifest.json", "p04", config, inputs=[str(p03 / "manifest.json")])
    model.to("cpu")
    if shuffled_model is not None:
        shuffled_model.to("cpu")
    release_device_cache(device)
