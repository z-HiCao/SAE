from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from sae_repro.data.cifar100 import IndexedCIFAR100


def save_top_image_grids(
    config: dict[str, Any],
    split: str,
    top_rows: torch.Tensor,
    latent_ids: list[int],
    output_dir: Path,
    activations: torch.Tensor | None = None,
) -> None:
    """为选中的高 MS latent 保存只含真实正激活样本的图像网格。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = IndexedCIFAR100(config, split)
    for latent_id in latent_ids:
        rows = [int(row) for row in top_rows[latent_id].tolist() if int(row) >= 0]
        if not rows:
            continue
        columns = 4
        row_count = (len(rows) + columns - 1) // columns
        figure, axes = plt.subplots(row_count, columns, figsize=(8, 2 * row_count))
        axes = np.asarray(axes, dtype=object).reshape(-1)
        for axis, row in zip(axes, rows):
            image, fine, coarse, sample_id = dataset[int(row)]
            axis.imshow(image)
            activation_text = ""
            if activations is not None:
                activation_text = f" a={float(activations[row, latent_id]):.4f}"
            axis.set_title(
                f"fine={fine} coarse={coarse}\nid={sample_id}{activation_text}",
                fontsize=7,
            )
            axis.axis("off")
        for axis in axes[len(rows) :]:
            axis.axis("off")
        figure.suptitle(f"SAE latent {latent_id} 的正激活图像（n={len(rows)}）")
        figure.tight_layout()
        figure.savefig(output_dir / f"latent_{latent_id:05d}.png", dpi=150)
        plt.close(figure)
