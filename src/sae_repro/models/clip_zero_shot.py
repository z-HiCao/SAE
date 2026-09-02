from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, CLIPModel

from sae_repro.core.device import release_device_cache


def _text_feature_tensor(output: Any) -> torch.Tensor:
    """兼容不同 transformers 版本的文本特征返回类型。"""
    if isinstance(output, torch.Tensor):
        return output
    for name in ("text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError("CLIP get_text_features 返回了无法识别的对象")


@torch.no_grad()
def build_clip_text_prototypes(
    model_name: str,
    class_names: list[str],
    templates: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """用提示词集构造归一化 CLIP 类别文本原型。"""
    if not templates:
        raise ValueError("至少需要一个 CLIP 文本提示模板")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    prompts = [
        template.format(label=name.replace("_", " "))
        for name in class_names
        for template in templates
    ]
    encoded: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        features = _text_feature_tensor(model.get_text_features(**tokens))
        encoded.append(F.normalize(features.float(), dim=-1).cpu())
    matrix = torch.cat(encoded).reshape(len(class_names), len(templates), -1)
    prototypes = F.normalize(matrix.mean(dim=1), dim=-1)
    logit_scale = float(model.logit_scale.exp().clamp(max=100).detach().cpu())
    model.to("cpu")
    del model
    release_device_cache(device)
    return prototypes, logit_scale


def clip_zero_shot_logits(
    standardized_embeddings: torch.Tensor,
    activation_mean: torch.Tensor,
    activation_std: torch.Tensor,
    text_prototypes: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    """把 SAE 标准化空间还原为 CLIP 空间并计算真实图文余弦 logits。"""
    raw = standardized_embeddings * activation_std + activation_mean
    normalized = F.normalize(raw.float(), dim=-1)
    prototypes = F.normalize(text_prototypes.float(), dim=-1)
    return float(logit_scale) * (normalized @ prototypes.T)
