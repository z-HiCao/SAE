from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paths import resolve_project_path


def stage_output_dir(config: dict[str, Any], stage: str) -> Path:
    """创建并返回某一阶段的稳定产物目录。"""
    root = resolve_project_path(config["paths"]["output_root"])
    path = root / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, producer: str) -> Path:
    """检查上游产物，缺失时明确指出应先运行哪个阶段。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少上游产物 {path}，请先运行 {producer}")
    return path


def save_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """以 UTF-8 和稳定缩进保存 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 字典。"""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"预期 JSON 字典：{path}")
    return data


def save_array(path: Path, array: np.ndarray) -> None:
    """保存连续 NumPy 数组并建立父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def write_manifest(
    path: Path,
    stage: str,
    config: dict[str, Any],
    status: str = "completed",
    inputs: list[str] | None = None,
) -> None:
    """记录阶段、环境、配置和上游输入，避免结果失去来源。"""
    payload = {
        "stage": stage,
        "status": status,
        "result_label": "ADAPTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "config": config,
        "inputs": inputs or [],
    }
    save_json(path, payload)

