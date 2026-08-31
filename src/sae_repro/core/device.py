from __future__ import annotations

import torch


def choose_device(requested: str = "auto") -> torch.device:
    """按 MPS、CUDA、CPU 的优先顺序选择设备。"""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release_device_cache(device: torch.device) -> None:
    """在模型切换阶段释放可回收的设备缓存。"""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

