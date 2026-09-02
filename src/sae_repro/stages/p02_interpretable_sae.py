from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sae_repro.core.artifacts import require_file, save_array, save_json, write_manifest
from sae_repro.core.device import choose_device
from sae_repro.core.preflight import ensure_code_analysis
from sae_repro.core.seed import seed_everything
from sae_repro.metrics.concepts import best_latent_per_concept, dictionary_match_scores
from sae_repro.metrics.disentanglement import (
    evaluate_unit_concept_mapping,
    latent_purity_rows,
    mapping_summary,
    one_to_one_dictionary_match,
    select_unit_concept_mapping,
    toy_concept_ablation,
)
from sae_repro.metrics.reconstruction import dead_latent_fraction, l0_score, r2_score
from sae_repro.sae.models import BatchTopKSAE, ReLUSAE
from sae_repro.sae.trainer import TrainingSpec, encode_and_reconstruct, train_sae

from .common import shared_array, stage_dir, tensor


def _candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """把配置中的扩展倍数、L1 和 TopK 组合展开为候选实验。"""
    expansions = [int(value) for value in config.get("expansion_factors", [config["expansion_factor"]])]
    specs: list[dict[str, Any]] = []
    for expansion in expansions:
        for coefficient in config.get("l1_coefficients", [config["l1_coefficient"]]):
            specs.append(
                {
                    "name": f"relu_l1_e{expansion}_lambda{float(coefficient):g}",
                    "model_type": "relu_l1",
                    "expansion_factor": expansion,
                    "l1_coefficient": float(coefficient),
                }
            )
        for k in config.get("topk_values", []):
            specs.append(
                {
                    "name": f"batch_topk_e{expansion}_k{int(k)}",
                    "model_type": "batch_topk",
                    "expansion_factor": expansion,
                    "k": int(k),
                }
            )
    if not specs:
        raise ValueError("P02 至少需要一个 SAE 候选配置")
    return specs


def _build_model(input_dim: int, spec: dict[str, Any]) -> torch.nn.Module:
    """按照候选配置构造 ReLU+L1 或 BatchTopK SAE。"""
    latent_dim = input_dim * int(spec["expansion_factor"])
    if spec["model_type"] == "relu_l1":
        return ReLUSAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            l1_coefficient=float(spec["l1_coefficient"]),
        )
    if spec["model_type"] == "batch_topk":
        return BatchTopKSAE(input_dim=input_dim, latent_dim=latent_dim, k=int(spec["k"]))
    raise ValueError(f"未知 P02 SAE 类型：{spec['model_type']}")


def _state_dict_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """复制一份 CPU checkpoint，避免后续候选训练覆盖最佳权重。"""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _candidate_is_better(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    config: dict[str, Any],
) -> bool:
    """优先在重建和稀疏约束内选择概念 F1 更高的候选。"""
    if current is None:
        return True
    minimum_r2 = float(config["selection_min_r2"])
    maximum_l0 = float(config["selection_max_l0"])

    def rank(row: dict[str, Any]) -> tuple[float, ...]:
        eligible = float(row["validation_r2"] >= minimum_r2 and row["validation_l0"] <= maximum_l0)
        r2_shortfall = max(0.0, minimum_r2 - float(row["validation_r2"]))
        l0_excess = max(0.0, float(row["validation_l0"]) - maximum_l0) / max(maximum_l0, 1.0)
        return (
            eligible,
            -r2_shortfall,
            -l0_excess,
            float(row["validation_mean_best_concept_f1"]),
            float(row["mean_ground_truth_direction_match"]),
            float(row["validation_r2"]),
            -float(row["validation_l0"]),
        )

    return rank(candidate) > rank(current)


def run(config: dict[str, Any]) -> None:
    """运行 P02 扫描，并用独立测试集比较 SAE latent 与原始 neuron。"""
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
    test_fine = torch.from_numpy(shared_array(config, "test", "fine")).long()
    test_coarse = torch.from_numpy(shared_array(config, "test", "coarse")).long()

    generator = torch.Generator().manual_seed(int(config["seed"]))
    permutation = torch.randperm(len(train_hidden), generator=generator)
    validation_size = max(1, round(len(permutation) * float(config["validation_fraction"])))
    validation_indices = permutation[:validation_size]
    fit_indices = permutation[validation_size:]
    if len(fit_indices) == 0:
        raise ValueError("P02 validation_fraction 过大，拟合集为空")

    fit_hidden = train_hidden[fit_indices]
    validation_hidden = train_hidden[validation_indices]
    validation_concepts = train_concepts[validation_indices]
    device = choose_device(str(config["device"]))
    candidate_rows: list[dict[str, Any]] = []
    selected_row: dict[str, Any] | None = None
    selected_spec: dict[str, Any] | None = None
    selected_state: dict[str, torch.Tensor] | None = None

    for candidate_index, spec in enumerate(_candidate_specs(config)):
        seed_everything(int(config["seed"]) + candidate_index)
        model = _build_model(train_hidden.shape[1], spec)
        history = train_sae(
            model,
            fit_hidden,
            TrainingSpec(
                steps=int(config["steps"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
            ),
            device,
        )
        validation_latents, validation_reconstruction = encode_and_reconstruct(
            model,
            validation_hidden,
            int(config["batch_size"]),
            device,
        )
        validation_mapping = best_latent_per_concept(
            validation_latents,
            validation_concepts,
            float(config["activation_threshold"]),
        )
        direction_match = dictionary_match_scores(
            model.decoder_directions().detach().cpu(),
            feature_directions,
        )
        row = {
            **spec,
            "validation_r2": r2_score(validation_hidden, validation_reconstruction),
            "validation_l0": l0_score(
                validation_latents,
                float(config["activation_threshold"]),
            ),
            "validation_dead_latent_fraction": dead_latent_fraction(
                validation_latents,
                float(config["activation_threshold"]),
            ),
            "validation_mean_best_concept_f1": float(
                np.mean([mapping["f1"] for mapping in validation_mapping])
            ),
            "mean_ground_truth_direction_match": float(direction_match.mean()),
            "final_train_loss": float(history[-1]["loss"]),
            "train_history": history,
        }
        candidate_rows.append(row)
        if _candidate_is_better(row, selected_row, config):
            selected_row = row
            selected_spec = spec
            selected_state = _state_dict_on_cpu(model)
        model.to("cpu")

    if selected_spec is None or selected_state is None or selected_row is None:
        raise RuntimeError("P02 没有成功选择 SAE 候选")
    selected_model = _build_model(train_hidden.shape[1], selected_spec)
    selected_model.load_state_dict(selected_state)
    selected_model.eval()

    train_latents, train_reconstruction = encode_and_reconstruct(
        selected_model,
        train_hidden,
        int(config["batch_size"]),
        device,
    )
    test_latents, test_reconstruction = encode_and_reconstruct(
        selected_model,
        test_hidden,
        int(config["batch_size"]),
        device,
    )
    activation_threshold = float(config["activation_threshold"])
    fixed_mapping = best_latent_per_concept(test_latents, test_concepts, activation_threshold)
    direction_match = dictionary_match_scores(
        selected_model.decoder_directions().detach().cpu(),
        feature_directions,
    )

    quantiles = [float(value) for value in config["interpretability_threshold_quantiles"]]
    raw_selection = select_unit_concept_mapping(
        train_hidden[fit_indices],
        train_hidden[validation_indices],
        validation_concepts,
        quantiles,
        allow_negative_direction=True,
    )
    raw_test_mapping = evaluate_unit_concept_mapping(test_hidden, test_concepts, raw_selection)
    sae_selection = select_unit_concept_mapping(
        train_latents[fit_indices],
        train_latents[validation_indices],
        validation_concepts,
        quantiles,
        allow_negative_direction=False,
    )
    sae_test_mapping = evaluate_unit_concept_mapping(test_latents, test_concepts, sae_selection)
    purity = latent_purity_rows(
        test_latents,
        test_fine,
        test_coarse,
        activation_threshold,
    )
    one_to_one_rows, one_to_one_summary = one_to_one_dictionary_match(
        selected_model.decoder_directions().detach().cpu(),
        feature_directions,
    )

    p01_checkpoint = torch.load(
        require_file(p01 / "toy_model.pt", "make p01"),
        map_location="cpu",
        weights_only=True,
    )
    p01_state = p01_checkpoint["state_dict"]
    interventions = toy_concept_ablation(
        selected_model,
        test_latents,
        test_concepts,
        sae_test_mapping,
        p01_state["weight"].float(),
        p01_state["bias"].float(),
        activation_threshold,
    )
    supported_purity = [
        row for row in purity if int(row["support"]) >= int(config["min_purity_support"])
    ]
    purity_summary = {
        "minimum_support": int(config["min_purity_support"]),
        "supported_latents": len(supported_purity),
        "mean_fine_purity": float(
            np.mean([row["fine_purity"] for row in supported_purity])
        )
        if supported_purity
        else 0.0,
        "mean_coarse_purity": float(
            np.mean([row["coarse_purity"] for row in supported_purity])
        )
        if supported_purity
        else 0.0,
        "mean_fine_entropy": float(
            np.mean([row["fine_entropy"] for row in supported_purity])
        )
        if supported_purity
        else 0.0,
    }

    comparison = {
        "raw_neuron": mapping_summary(raw_test_mapping),
        "sae_latent": mapping_summary(sae_test_mapping),
        "interpretability_protocol": (
            "所有阈值仅由 P01 train 的拟合/验证划分选择，最终 F1 在独立 test 上计算"
        ),
    }
    metrics = {
        "status": "ADAPTED",
        "selected_candidate": selected_row,
        "test_r2": r2_score(test_hidden, test_reconstruction),
        "test_l0": l0_score(test_latents, activation_threshold),
        "dead_latent_fraction": dead_latent_fraction(test_latents, activation_threshold),
        "mean_best_concept_f1": float(np.mean([row["f1"] for row in fixed_mapping])),
        "mean_ground_truth_direction_match": float(direction_match.mean()),
        "raw_vs_sae_interpretability": comparison,
        "latent_purity_summary": purity_summary,
        "one_to_one_direction_match": one_to_one_summary,
        "concept_mapping": fixed_mapping,
        "evidence_boundary": (
            "只有同时满足高重建、低 L0、SAE 优于 raw neuron、较高 purity 和一对一方向恢复，"
            "才能把结果解释为成功拆解 superposition"
        ),
    }
    checkpoint: dict[str, Any] = {
        "state_dict": selected_model.to("cpu").state_dict(),
        "model_type": selected_spec["model_type"],
        "input_dim": int(train_hidden.shape[1]),
        "latent_dim": int(train_latents.shape[1]),
        "expansion_factor": int(selected_spec["expansion_factor"]),
    }
    if selected_spec["model_type"] == "relu_l1":
        checkpoint["l1_coefficient"] = float(selected_spec["l1_coefficient"])
    else:
        checkpoint["k"] = int(selected_spec["k"])

    save_array(output / "train_activations.npy", train_hidden.numpy())
    save_array(output / "test_activations.npy", test_hidden.numpy())
    save_array(output / "train_latents.npy", train_latents.numpy())
    save_array(output / "test_latents.npy", test_latents.numpy())
    save_array(
        output / "decoder_directions.npy",
        selected_model.decoder_directions().detach().cpu().numpy(),
    )
    save_array(output / "direction_match.npy", direction_match.numpy())
    torch.save(checkpoint, output / "sae.pt")
    save_json(output / "metrics.json", metrics)
    save_json(output / "sweep_metrics.json", {"candidates": candidate_rows})
    save_json(
        output / "interpretability_comparison.json",
        {
            **comparison,
            "raw_mapping": raw_test_mapping,
            "sae_mapping": sae_test_mapping,
        },
    )
    save_json(
        output / "disentanglement_metrics.json",
        {
            "latent_purity_summary": purity_summary,
            "latent_purity": purity,
            "one_to_one_summary": one_to_one_summary,
            "one_to_one_mapping": one_to_one_rows,
        },
    )
    save_json(output / "toy_interventions.json", {"ablation_results": interventions})
    write_manifest(output / "manifest.json", "p02", config, inputs=[str(p01 / "manifest.json")])
