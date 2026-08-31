from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import require_file, save_array, save_json, write_manifest
from sae_repro.core.device import choose_device
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.concepts import best_latent_per_concept, dictionary_match_scores
from sae_repro.metrics.reconstruction import dead_latent_fraction, l0_score, r2_score
from sae_repro.sae.models import ReLUSAE
from sae_repro.sae.trainer import TrainingSpec, encode_and_reconstruct, train_sae

from .common import shared_array, stage_dir, tensor


def run(config: dict[str, Any]) -> None:
    """运行第二阶段，从 P01 的叠加瓶颈中恢复稀疏 feature。"""
    ensure_code_analysis(config)
    seed_everything(int(config["seed"]))
    p01 = stage_dir(config, "p01")
    output = stage_dir(config, "p02")
    train_hidden = tensor(np.load(require_file(p01 / "train_hidden.npy", "make p01")))
    test_hidden = tensor(np.load(require_file(p01 / "test_hidden.npy", "make p01")))
    feature_directions = tensor(
        np.load(require_file(p01 / "feature_directions.npy", "make p01"))
    )
    train_concepts = tensor(shared_array(config, "train", "concepts"))
    test_concepts = tensor(shared_array(config, "test", "concepts"))
    latent_dim = train_hidden.shape[1] * int(config["expansion_factor"])
    model = ReLUSAE(
        input_dim=train_hidden.shape[1],
        latent_dim=latent_dim,
        l1_coefficient=float(config["l1_coefficient"]),
    )
    device = choose_device(str(config["device"]))
    history = train_sae(
        model,
        train_hidden,
        TrainingSpec(
            steps=int(config["steps"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
        ),
        device,
    )
    train_latents, train_reconstruction = encode_and_reconstruct(
        model, train_hidden, int(config["batch_size"]), device
    )
    test_latents, test_reconstruction = encode_and_reconstruct(
        model, test_hidden, int(config["batch_size"]), device
    )
    mapping = best_latent_per_concept(
        test_latents,
        test_concepts,
        float(config["activation_threshold"]),
    )
    match = dictionary_match_scores(model.decoder_directions().detach().cpu(), feature_directions)
    metrics = {
        "status": "ADAPTED",
        "test_r2": r2_score(test_hidden, test_reconstruction),
        "test_l0": l0_score(test_latents, float(config["activation_threshold"])),
        "dead_latent_fraction": dead_latent_fraction(
            test_latents, float(config["activation_threshold"])
        ),
        "mean_best_concept_f1": float(np.mean([row["f1"] for row in mapping])),
        "mean_ground_truth_direction_match": float(match.mean()),
        "train_history": history,
        "concept_mapping": mapping,
    }
    save_array(output / "train_activations.npy", train_hidden.numpy())
    save_array(output / "test_activations.npy", test_hidden.numpy())
    save_array(output / "train_latents.npy", train_latents.numpy())
    save_array(output / "test_latents.npy", test_latents.numpy())
    save_array(
        output / "decoder_directions.npy", model.decoder_directions().detach().cpu().numpy()
    )
    save_array(output / "direction_match.npy", match.numpy())
    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "l1_coefficient": model.l1_coefficient,
        },
        output / "sae.pt",
    )
    save_json(output / "metrics.json", metrics)
    write_manifest(output / "manifest.json", "p02", config, inputs=[str(p01 / "manifest.json")])

