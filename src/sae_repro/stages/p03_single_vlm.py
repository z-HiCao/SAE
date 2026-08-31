from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import load_json, require_file, save_array, save_json, write_manifest
from sae_repro.core.device import choose_device
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.concepts import (
    class_centroid_weights,
    latent_target_classes,
    top_activating_indices,
)
from sae_repro.metrics.monosemanticity import monosemanticity_scores
from sae_repro.metrics.reconstruction import dead_latent_fraction, l0_score, r2_score
from sae_repro.models.vision import extract_vision_embeddings
from sae_repro.sae.models import MatryoshkaBatchTopKSAE
from sae_repro.sae.trainer import TrainingSpec, encode_and_reconstruct, train_sae
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


@torch.no_grad()
def _intervention_effects(
    model: MatryoshkaBatchTopKSAE,
    train_latents: torch.Tensor,
    test_latents: torch.Tensor,
    test_reconstruction: torch.Tensor,
    train_labels: torch.Tensor,
    train_activations: torch.Tensor,
    ms_scores: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, float | int]]:
    """用 class centroid probe 评价单 latent 正向钳制的目标 logit 变化。"""
    model = model.to("cpu").eval()
    weights = class_centroid_weights(train_activations, train_labels, classes=100)
    targets = latent_target_classes(train_latents, train_labels, classes=100)
    valid = torch.where(torch.isfinite(ms_scores))[0]
    order = valid[torch.argsort(ms_scores[valid], descending=True)]
    order = order[: int(config["intervention_latents"])]
    rows: list[dict[str, float | int]] = []
    for latent_id in order.tolist():
        target = int(targets[latent_id])
        value = torch.quantile(train_latents[:, latent_id], float(config["intervention_quantile"]))
        modified = test_latents.clone()
        modified[:, latent_id] = torch.maximum(modified[:, latent_id], value)
        reconstruction = model.decode(modified)
        direction = weights[target]
        delta = ((reconstruction - test_reconstruction) @ direction).mean()
        rows.append(
            {
                "latent_id": int(latent_id),
                "target_fine_class": target,
                "ms": float(ms_scores[latent_id]),
                "clamp_value": float(value),
                "mean_target_probe_logit_delta": float(delta),
            }
        )
    return rows


def run(config: dict[str, Any]) -> None:
    """运行主体论文的单 VLM 最小链路。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    p02 = stage_dir(config, "p02")
    require_file(p02 / "manifest.json", "make p02")
    p02_checkpoint_path = require_file(p02 / "sae.pt", "make p02")
    p02_checkpoint = torch.load(p02_checkpoint_path, map_location="cpu", weights_only=True)
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
        model, train_tensor, int(config["sae_batch_size"]), device
    )
    test_latents, test_reconstruction = encode_and_reconstruct(
        model, test_tensor, int(config["sae_batch_size"]), device
    )
    test_concepts = tensor(shared_array(config, "test", "concepts"))
    train_labels = torch.from_numpy(shared_array(config, "train", "fine")).long()
    ms_latent = monosemanticity_scores(test_latents, test_concepts)
    ms_raw = monosemanticity_scores(test_tensor, test_concepts)
    top_indices = top_activating_indices(test_latents, int(config["top_images"]))
    valid_visual = torch.where(torch.isfinite(ms_latent))[0]
    visual_order = valid_visual[torch.argsort(ms_latent[valid_visual], descending=True)]
    save_top_image_grids(
        config=config,
        split="test",
        top_rows=top_indices,
        latent_ids=visual_order[: int(config["visualized_latents"])].tolist(),
        output_dir=output / "top_image_grids",
    )
    effects = _intervention_effects(
        model,
        train_latents,
        test_latents,
        test_reconstruction,
        train_labels,
        train_tensor,
        ms_latent,
        config,
    )
    finite_latent_ms = ms_latent[torch.isfinite(ms_latent)]
    finite_raw_ms = ms_raw[torch.isfinite(ms_raw)]
    metrics = {
        "status": "ADAPTED",
        "test_r2": r2_score(test_tensor, test_reconstruction),
        "test_l0": l0_score(test_latents),
        "dead_latent_fraction": dead_latent_fraction(test_latents),
        "best_latent_ms": float(finite_latent_ms.max()) if len(finite_latent_ms) else float("nan"),
        "mean_latent_ms": float(finite_latent_ms.mean()) if len(finite_latent_ms) else float("nan"),
        "best_raw_neuron_ms": float(finite_raw_ms.max()) if len(finite_raw_ms) else float("nan"),
        "mean_raw_neuron_ms": float(finite_raw_ms.mean()) if len(finite_raw_ms) else float("nan"),
        "intervention_effects": effects,
        "train_history": history,
        "causal_boundary": "只评价 SAE 重建激活上的 class centroid probe，不等同于 LLaVA 生成干预",
        "upstream_p02_expansion_factor": upstream_expansion,
        "upstream_p02_mean_best_concept_f1": p02_metrics["mean_best_concept_f1"],
    }
    save_array(output / "train_latents.npy", train_latents.numpy())
    save_array(output / "test_latents.npy", test_latents.numpy())
    save_array(output / "train_reconstruction.npy", train_reconstruction.numpy())
    save_array(output / "test_reconstruction.npy", test_reconstruction.numpy())
    save_array(output / "ms_latent.npy", ms_latent.numpy())
    save_array(output / "ms_raw.npy", ms_raw.numpy())
    save_array(output / "top_image_rows.npy", top_indices.numpy())
    save_array(
        output / "decoder_directions.npy", model.decoder_directions().detach().cpu().numpy()
    )
    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "k": model.k,
            "fractions": config["matryoshka_fractions"],
        },
        output / "clip_sae.pt",
    )
    save_json(output / "metrics.json", metrics)
    write_manifest(
        output / "manifest.json",
        "p03",
        config,
        inputs=[str(stage_dir(config, "p02") / "manifest.json")],
    )
