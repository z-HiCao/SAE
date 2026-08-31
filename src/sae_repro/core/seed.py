from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

