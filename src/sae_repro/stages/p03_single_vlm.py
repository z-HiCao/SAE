from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import load_json, require_file, save_array, save_json, write_manifest
from sae_repro.core.device import choose_device, release_device_cache
from sae_repro.core.paths import resolve_project_path
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.concepts import positive_top_activating_indices
from sae_repro.metrics.interventions import evaluate_clip_interventions
from sae_repro.metrics.monosemanticity import (
    monosemanticity_scores,
    monosemanticity_subsample_intervals,
    supported_monosemanticity_scores,
)
from sae_repro.metrics.reconstruction import dead_latent_fraction, l0_score, r2_score
from sae_repro.models.clip_zero_shot import build_clip_text_prototypes, clip_zero_shot_logits
from sae_repro.models.vision import extract_vision_embeddings
from sae_repro.sae.models import MatryoshkaBatchTopKSAE
from sae_repro.sae.trainer import TrainingSpec, encode_and_reconstruct, train_sae
from sae_repro.visualization.interventions import save_intervention_plots
from sae_repro.visualization.top_images import save_top_image_grids

from .common import shared_array, stage_dir, standardize_train_test, tensor


def _load_or_extract(
    config: dict[str, Any],
    output: Any,
    split: str,
) -> np.ndarray:
    """复用已缓存的 CLIP 激活，否则按共享索引提取。"""
    path = output / f"{split}_clip_raw.npy"
    id_path = output / f"{split}_sample_ids.npy"
    if path.exists() and id_path.exists():
        return np.load(path)
    return extract_vision_embeddings(
        config=config,
        split=split,
        model_name=str(config["model_name"]),
        model_kind=str(config["model_kind"]),
        batch_size=int(config["model_batch_size"]),
        output_path=path,
        id_path=id_path,
    )


def _semantic_reference(
    config: dict[str, Any],
    output: Any,
) -> tuple[torch.Tensor, str]:
    """读取标签语义，或用独立视觉模型提取测试图片语义嵌入。"""
    reference_kind = str(config.get("semantic_reference_kind", "labels"))
    if reference_kind == "labels":
        return tensor(shared_array(config, "test", "concepts")), "cifar100_hierarchical_labels"
    path = output / "test_semantic_reference_raw.npy"
    id_path = output / "test_semantic_reference_sample_ids.npy"
    if path.exists() and id_path.exists():
        matrix = np.load(path)
    else:
        matrix = extract_vision_embeddings(
            config=config,
            split="test",
            model_name=str(config["semantic_model_name"]),
            model_kind=reference_kind,
            batch_size=int(config["semantic_model_batch_size"]),
            output_path=path,
            id_path=id_path,
        )
    cached_ids = np.load(require_file(id_path, "重新提取独立语义嵌入"))
    expected_ids = shared_array(config, "test", "indices")
    if not np.array_equal(cached_ids, expected_ids):
        raise RuntimeError("独立语义嵌入与共享 CIFAR-100 测试样本顺序不一致")
    return tensor(matrix), f"{reference_kind}:{config['semantic_model_name']}"


def _finite_mean(values: torch.Tensor) -> float:
    """只在有限值上计算均值，没有有限值时返回 NaN。"""
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _finite_max(values: torch.Tensor) -> float:
    """只在有限值上计算最大值，没有有限值时返回 NaN。"""
    finite = values[torch.isfinite(values)]
    return float(finite.max()) if finite.numel() else float("nan")


def run(config: dict[str, Any]) -> None:
    """运行主体论文的单 VLM SAE、稳健 MS 与 CLIP 因果干预链路。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    p02 = stage_dir(config, "p02")
    require_file(p02 / "manifest.json", "make p02")
    p02_checkpoint = torch.load(
        require_file(p02 / "sae.pt", "make p02"),
        map_location="cpu",
        weights_only=True,
    )
    p02_metrics = load_json(require_file(p02 / "metrics.json", "make p02"))
    output = stage_dir(config, "p03")
    train_raw = _load_or_extract(config, output, "train")
    test_raw = _load_or_extract(config, output, "test")
    for split in ("train", "test"):
        cached_ids = np.load(require_file(output / f"{split}_sample_ids.npy", "重新提取 CLIP 激活"))
        expected_ids = shared_array(config, split, "indices")
        if not np.array_equal(cached_ids, expected_ids):
            raise RuntimeError(f"CLIP {split} 激活与共享 CIFAR-100 样本顺序不一致")

    train, test, mean, std = standardize_train_test(train_raw, test_raw)
    save_array(output / "train_activations.npy", train)
    save_array(output / "test_activations.npy", test)
    save_array(output / "activation_mean.npy", mean)
    save_array(output / "activation_std.npy", std)
    train_tensor = tensor(train)
    test_tensor = tensor(test)
    upstream_expansion = int(p02_checkpoint["latent_dim"]) // int(p02_checkpoint["input_dim"])
    expansion_factor = (
        upstream_expansion
        if bool(config.get("reuse_p02_expansion", False))
        else int(config["sae_expansion_factor"])
    )
    latent_dim = train_tensor.shape[1] * expansion_factor
    model = MatryoshkaBatchTopKSAE(
        input_dim=train_tensor.shape[1],
        latent_dim=latent_dim,
        k=int(config["sae_k"]),
        fractions=[float(value) for value in config["matryoshka_fractions"]],
    )
    device = choose_device(str(config["device"]))
    history = train_sae(
        model,
        train_tensor,
        TrainingSpec(
            steps=int(config["sae_steps"]),
            batch_size=int(config["sae_batch_size"]),
            learning_rate=float(config["learning_rate"]),
        ),
        device,
    )
    train_latents, train_reconstruction = encode_and_reconstruct(
        model,
        train_tensor,
        int(config["sae_batch_size"]),
        device,
    )
    test_latents, test_reconstruction = encode_and_reconstruct(
        model,
        test_tensor,
        int(config["sae_batch_size"]),
        device,
    )
    decoder_directions = model.decoder_directions().detach().cpu()
    model.to("cpu")
    release_device_cache(device)

    semantic_embeddings, semantic_source = _semantic_reference(config, output)
    ms_latent_all = monosemanticity_scores(test_latents, semantic_embeddings)
    ms_latent_supported, support = supported_monosemanticity_scores(
        test_latents,
        semantic_embeddings,
        minimum_support=int(config["minimum_ms_support"]),
        activation_threshold=float(config["activation_threshold"]),
    )
    ms_raw = monosemanticity_scores(test_tensor, semantic_embeddings)
    top_indices, top_positive_counts = positive_top_activating_indices(
        test_latents,
        int(config["top_images"]),
        float(config["activation_threshold"]),
    )
    valid_visual = torch.where(torch.isfinite(ms_latent_supported))[0]
    visual_order = valid_visual[
        torch.argsort(ms_latent_supported[valid_visual], descending=True)
    ]
    visual_ids = visual_order[: int(config["visualized_latents"])].tolist()
    save_top_image_grids(
        config=config,
        split="test",
        top_rows=top_indices,
        latent_ids=visual_ids,
        output_dir=output / "top_positive_image_grids",
        activations=test_latents,
    )
    interval_ids = visual_order[: int(config["ms_interval_latents"])].tolist()
    intervals = monosemanticity_subsample_intervals(
        test_latents,
        semantic_embeddings,
        interval_ids,
        repeats=int(config["ms_interval_repeats"]),
        fraction=float(config["ms_interval_fraction"]),
        seed=int(config["seed"]),
    )

    shared_manifest = load_json(
        require_file(
            resolve_project_path(config["paths"]["shared_root"]) / "manifest.json",
            "make prepare",
        )
    )
    fine_names = [str(name) for name in shared_manifest["fine_names"]]
    text_prototypes, logit_scale = build_clip_text_prototypes(
        str(config["model_name"]),
        fine_names,
        [str(template) for template in config["clip_prompt_templates"]],
        int(config["text_batch_size"]),
        device,
    )
    train_labels = torch.from_numpy(shared_array(config, "train", "fine")).long()
    test_labels = torch.from_numpy(shared_array(config, "test", "fine")).long()
    intervention_rows, intervention_summary = evaluate_clip_interventions(
        train_latents=train_latents,
        test_latents=test_latents,
        test_reconstruction=test_reconstruction,
        decoder_directions=decoder_directions,
        train_labels=train_labels,
        test_labels=test_labels,
        ms_scores=ms_latent_supported,
        activation_mean=tensor(mean),
        activation_std=tensor(std),
        text_prototypes=text_prototypes,
        logit_scale=logit_scale,
        intervention_latents=int(config["intervention_latents"]),
        minimum_support=int(config["minimum_intervention_support"]),
        quantile=float(config["intervention_positive_quantile"]),
        doses=[float(value) for value in config["intervention_doses"]],
        activation_threshold=float(config["activation_threshold"]),
    )
    original_logits = clip_zero_shot_logits(
        test_tensor,
        tensor(mean),
        tensor(std),
        text_prototypes,
        logit_scale,
    )
    reconstruction_logits = clip_zero_shot_logits(
        test_reconstruction,
        tensor(mean),
        tensor(std),
        text_prototypes,
        logit_scale,
    )
    original_accuracy = float((original_logits.argmax(dim=1) == test_labels).float().mean())
    reconstruction_accuracy = float(
        (reconstruction_logits.argmax(dim=1) == test_labels).float().mean()
    )
    save_intervention_plots(
        intervention_rows,
        fine_names,
        output / "intervention_plots",
    )

    finite_all = int(torch.isfinite(ms_latent_all).sum())
    finite_supported = int(torch.isfinite(ms_latent_supported).sum())
    metrics = {
        "status": "ADAPTED",
        "test_r2": r2_score(test_tensor, test_reconstruction),
        "test_l0": l0_score(test_latents, float(config["activation_threshold"])),
        "dead_latent_fraction": dead_latent_fraction(
            test_latents,
            float(config["activation_threshold"]),
        ),
        "semantic_reference_source": semantic_source,
        "minimum_ms_support": int(config["minimum_ms_support"]),
        "finite_latent_ms_without_support_filter": finite_all,
        "supported_latent_ms_count": finite_supported,
        "supported_latent_ms_coverage": finite_supported / max(len(ms_latent_supported), 1),
        "best_latent_ms": _finite_max(ms_latent_supported),
        "mean_latent_ms": _finite_mean(ms_latent_supported),
        "best_raw_neuron_ms": _finite_max(ms_raw),
        "mean_raw_neuron_ms": _finite_mean(ms_raw),
        "ms_subsample_intervals": intervals,
        "clip_zero_shot_original_accuracy": original_accuracy,
        "clip_zero_shot_reconstruction_accuracy": reconstruction_accuracy,
        "intervention_summary": intervention_summary,
        "intervention_effects": intervention_rows,
        "train_history": history,
        "upstream_p02_expansion_factor": upstream_expansion,
        "upstream_p02_mean_best_concept_f1": p02_metrics["mean_best_concept_f1"],
        "evidence_boundary": (
            "支持度过滤和独立语义参考降低了小样本高 MS 偏差；CLIP logits 干预仍不是原论文的"
            "中间 token 与 LLaVA 生成干预"
        ),
    }
    save_array(output / "train_latents.npy", train_latents.numpy())
    save_array(output / "test_latents.npy", test_latents.numpy())
    save_array(output / "train_reconstruction.npy", train_reconstruction.numpy())
    save_array(output / "test_reconstruction.npy", test_reconstruction.numpy())
    save_array(output / "decoder_directions.npy", decoder_directions.numpy())
    save_array(output / "ms_latent.npy", ms_latent_all.numpy())
    save_array(output / "ms_latent_supported.npy", ms_latent_supported.numpy())
    save_array(output / "ms_support_counts.npy", support.numpy())
    save_array(output / "ms_raw.npy", ms_raw.numpy())
    save_array(output / "top_image_rows.npy", top_indices.numpy())
    save_array(output / "top_positive_counts.npy", top_positive_counts.numpy())
    save_array(output / "clip_text_prototypes.npy", text_prototypes.numpy())
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "k": model.k,
            "fractions": config["matryoshka_fractions"],
        },
        output / "clip_sae.pt",
    )
    save_json(output / "intervention_results.json", {
        "summary": intervention_summary,
        "results": intervention_rows,
    })
    save_json(output / "metrics.json", metrics)
    write_manifest(
        output / "manifest.json",
        "p03",
        config,
        inputs=[str(p02 / "manifest.json")],
    )
