from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import require_file, save_array, save_json, write_manifest
from sae_repro.core.device import choose_device
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.universality import (
    cofire_proportion,
    concept_energy,
    cross_reconstruction_matrix,
    firing_entropy,
)
from sae_repro.models.vision import extract_vision_embeddings
from sae_repro.sae.models import UniversalSAE
from sae_repro.sae.trainer import train_universal_sae

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

    model = UniversalSAE(
        input_dims=[clip_train.shape[1], siglip_train_tensor.shape[1]],
        latent_dim=int(config["latent_dim"]),
        k=int(config["k"]),
    )
    device = choose_device(str(config["device"]))
    history = train_universal_sae(
        model,
        [clip_train, siglip_train_tensor],
        steps=int(config["steps"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        loss_name=str(config["reconstruction_loss"]),
        device=device,
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
    entropy = firing_entropy(test_latents)
    cofire = cofire_proportion(test_latents)
    energy_clip = concept_energy(
        train_latents[0], model.decoder_directions(0).detach().cpu()
    )
    energy_siglip = concept_energy(
        train_latents[1], model.decoder_directions(1).detach().cpu()
    )
    metrics = {
        "status": "ADAPTED",
        "cross_reconstruction_r2": cross_r2.tolist(),
        "mean_firing_entropy": float(torch.nanmean(entropy)),
        "mean_cofire_clip": float(cofire[0].mean()),
        "mean_cofire_siglip": float(cofire[1].mean()),
        "mean_energy_clip": float(energy_clip.mean()),
        "mean_energy_siglip": float(energy_siglip.mean()),
        "train_history": history,
        "semantic_boundary": "共享 latent 索引由训练目标约束，不能仅凭同索引断言语义完全相同",
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
    save_json(output / "metrics.json", metrics)
    write_manifest(output / "manifest.json", "p04", config, inputs=[str(p03 / "manifest.json")])
