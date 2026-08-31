from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, SiglipVisionModel

from sae_repro.core.artifacts import save_array
from sae_repro.core.device import choose_device, release_device_cache
from sae_repro.data.cifar100 import IndexedCIFAR100


def _collate_images(batch: list[tuple[Any, int, int, int]]) -> tuple[list[Any], np.ndarray]:
    """保留 PIL 图像列表，并只返回共享 sample id。"""
    images = [row[0] for row in batch]
    sample_ids = np.asarray([row[3] for row in batch], dtype=np.int64)
    return images, sample_ids


def _load_vision_model(model_name: str, model_kind: str) -> tuple[Any, torch.nn.Module]:
    """按明确类型加载视觉塔，避免同时载入不需要的文本塔。"""
    processor = AutoImageProcessor.from_pretrained(model_name)
    if model_kind == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    elif model_kind == "siglip":
        model = SiglipVisionModel.from_pretrained(model_name)
    else:
        raise ValueError(f"当前仅支持 clip 或 siglip，收到：{model_kind}")
    return processor, model


def _select_embedding(output: Any, model_kind: str) -> torch.Tensor:
    """为不同模型选择固定长度的图像级激活。"""
    if model_kind == "clip":
        return output.image_embeds
    if output.pooler_output is not None:
        return output.pooler_output
    return output.last_hidden_state.mean(dim=1)


def extract_vision_embeddings(
    config: dict[str, Any],
    split: str,
    model_name: str,
    model_kind: str,
    batch_size: int,
    output_path: Path,
    id_path: Path,
) -> np.ndarray:
    """按共享 CIFAR 索引提取图像级 VLM 激活并立即写盘。"""
    dataset = IndexedCIFAR100(config, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        collate_fn=_collate_images,
    )
    device = choose_device(str(config.get("device", "auto")))
    processor, model = _load_vision_model(model_name, model_kind)
    model.eval().to(device)
    activations: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    with torch.inference_mode():
        for images, ids in tqdm(loader, desc=f"提取 {model_kind}-{split} 激活"):
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            output = model(pixel_values=pixel_values)
            embedding = _select_embedding(output, model_kind)
            activations.append(embedding.float().cpu().numpy())
            sample_ids.append(ids)
    matrix = np.concatenate(activations, axis=0).astype(np.float32, copy=False)
    ids = np.concatenate(sample_ids, axis=0)
    if len(matrix) != len(dataset):
        raise RuntimeError("激活数量与共享样本数量不一致")
    save_array(output_path, matrix)
    save_array(id_path, ids)
    model.to("cpu")
    del model
    release_device_cache(device)
    return matrix

