from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_absorption_plots(
    reports_by_source: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """保存每个父概念的留出集 absorption 与匹配随机对照。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, reports in reports_by_source.items():
        if not reports:
            continue
        parent_ids = [int(row["parent_id"]) for row in reports]
        observed = [float(row["absorption_rate_over_tested_false_negatives"]) for row in reports]
        null_mean = [float(row["matched_random_control"]["null_mean"]) for row in reports]
        positions = np.arange(len(parent_ids))
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.bar(positions - 0.2, observed, width=0.4, label="固定候选消融")
        axis.bar(positions + 0.2, null_mean, width=0.4, label="匹配随机候选")
        axis.set_xticks(positions, [str(value) for value in parent_ids])
        axis.set_xlabel("coarse parent ID")
        axis.set_ylabel("测试漏检样本中的 absorption 比例")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{source}：留出测试集 absorption 与零假设")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{source}_absorption_vs_null.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        for row in reports:
            curve = row["splitting"]["union_f1_curve"]
            axis.plot(range(1, len(curve) + 1), curve, alpha=0.65)
        axis.set_xlabel("按发现集排序后加入的 latent 数量")
        axis.set_ylabel("测试集并集 F1")
        axis.set_title(f"{source}：固定 latent 顺序的 splitting 曲线")
        figure.tight_layout()
        figure.savefig(output_dir / f"{source}_splitting_curves.png", dpi=180)
        plt.close(figure)
