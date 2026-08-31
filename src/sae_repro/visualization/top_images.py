from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from sae_repro.data.cifar100 import IndexedCIFAR100


def save_top_image_grids(
    config: dict[str, Any],
    split: str,
    top_rows: torch.Tensor,
    latent_ids: list[int],
    output_dir: Path,
) -> None:
    """为选中的高 MS latent 保存 top activating CIFAR 图像网格。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = IndexedCIFAR100(config, split)
    for latent_id in latent_ids:
        rows = top_rows[latent_id].tolist()
        columns = 4
        row_count = (len(rows) + columns - 1) // columns
        figure, axes = plt.subplots(row_count, columns, figsize=(8, 2 * row_count))
        axes = axes.reshape(-1)
        for axis, row in zip(axes, rows):
            image, fine, coarse, sample_id = dataset[int(row)]
            axis.imshow(image)
            axis.set_title(f"fine={fine} coarse={coarse}\nid={sample_id}", fontsize=7)
            axis.axis("off")
        for axis in axes[len(rows) :]:
            axis.axis("off")
        figure.suptitle(f"SAE latent {latent_id} 的 top activating images")
        figure.tight_layout()
        figure.savefig(output_dir / f"latent_{latent_id:05d}.png", dpi=150)
        plt.close(figure)

