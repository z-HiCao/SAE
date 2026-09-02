from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_cross_model_alignment_plots(
    rows: list[dict[str, Any]],
    null_controls: dict[str, Any],
    output_dir: Path,
) -> None:
    """保存 latent 级对齐散点图和观察值/零假设对比图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    supported = [row for row in rows if row["supported_in_both"]]
    if supported:
        x = np.asarray([row["activation_pearson"] for row in supported], dtype=float)
        y = np.asarray([row["fine_profile_cosine"] for row in supported], dtype=float)
        color = np.asarray(
            [min(row["support_first"], row["support_second"]) for row in supported],
            dtype=float,
        )
        finite = np.isfinite(x) & np.isfinite(y)
        figure, axis = plt.subplots(figsize=(7, 5))
        points = axis.scatter(x[finite], y[finite], c=color[finite], s=18, alpha=0.75)
        axis.set_xlabel("同索引 latent 激活 Pearson 相关")
        axis.set_ylabel("fine 标签分布余弦相似度")
        axis.set_title("CLIP 与 SigLIP 的共享 latent 对齐")
        figure.colorbar(points, ax=axis, label="两个模型中的较小支持数")
        figure.tight_layout()
        figure.savefig(output_dir / "latent_alignment_scatter.png", dpi=180)
        plt.close(figure)

    comparisons: list[tuple[str, float, float]] = []
    for family in ("sample_permutation", "latent_permutation"):
        for name, values in null_controls.get(family, {}).items():
            comparisons.append((name, float(values["observed"]), float(values["null_mean"])))
    if comparisons:
        names = [item[0] for item in comparisons]
        observed = [item[1] for item in comparisons]
        null_mean = [item[2] for item in comparisons]
        positions = np.arange(len(names))
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(positions - 0.18, observed, width=0.36, label="观察值")
        axis.bar(positions + 0.18, null_mean, width=0.36, label="置换零假设均值")
        axis.set_xticks(positions, names, rotation=20, ha="right")
        axis.set_ylabel("指标值")
        axis.set_title("跨模型对齐与置换对照")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "alignment_null_comparison.png", dpi=180)
        plt.close(figure)
