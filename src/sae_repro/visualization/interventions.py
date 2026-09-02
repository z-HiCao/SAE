from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_intervention_plots(
    results: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """保存总体剂量响应和逐 latent 消融效果图。"""
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    doses = [float(row["dose"]) for row in results[0]["dose_response"]]
    target_matrix = np.asarray(
        [
            [float(dose_row["mean_target_logit_delta"]) for dose_row in row["dose_response"]]
            for row in results
        ]
    )
    control_matrix = np.asarray(
        [
            [float(dose_row["matched_control_target_delta"]) for dose_row in row["dose_response"]]
            for row in results
        ]
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(doses, target_matrix.mean(axis=0), marker="o", label="目标 latent")
    axis.plot(doses, control_matrix.mean(axis=0), marker="o", label="匹配对照 latent")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("注入剂量倍数")
    axis.set_ylabel("目标 CLIP logit 平均变化")
    axis.set_title("SAE latent 注入的总体剂量响应")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "dose_response.png", dpi=160)
    plt.close(figure)

    ordered = sorted(
        results,
        key=lambda row: abs(float(row["ablation_mean_target_logit_delta"])),
        reverse=True,
    )[:20]
    labels = [
        f"z{int(row['latent_id'])}:{class_names[int(row['target_fine_class'])]}"
        for row in ordered
    ]
    values = [float(row["ablation_mean_target_logit_delta"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(ordered))))
    positions = np.arange(len(ordered))
    axis.barh(positions, values)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("消融后的目标 CLIP logit 变化")
    axis.set_title("单 latent 消融效果")
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_effects.png", dpi=160)
    plt.close(figure)
