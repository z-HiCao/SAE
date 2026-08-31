from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import project_root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置，阶段配置覆盖公共配置。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取一个 YAML 文件并校验顶层对象。"""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"配置顶层必须是字典：{path}")
    return data


def load_config(stage: str | None = None) -> dict[str, Any]:
    """读取公共配置，并按需合并某个论文阶段配置。"""
    config_dir = project_root() / "configs"
    config = _read_yaml(config_dir / "base.yaml")
    if stage is not None:
        stage_path = config_dir / f"{stage}.yaml"
        if not stage_path.exists():
            raise FileNotFoundError(f"找不到阶段配置：{stage_path}")
        config = _deep_merge(config, _read_yaml(stage_path))
    return config

