from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import project_root


REQUIRED_ANALYSIS_FILES = (
    "repository_tree.md",
    "module_inventory.csv",
    "equation_code_map.csv",
    "call_graph.mmd",
    "tensor_shapes.md",
    "assumptions.yaml",
    "CODE_ANALYSIS_DONE.yaml",
)


def ensure_code_analysis(config: dict[str, Any]) -> None:
    """正式运行前检查代码结构分析是否存在并已接受。"""
    if config.get("allow_unreviewed_code", False):
        return
    analysis_dir = project_root() / "docs" / "code_structure"
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not (analysis_dir / name).exists()]
    if missing:
        raise RuntimeError(f"代码结构分析不完整，缺少：{', '.join(missing)}")
    status_path = analysis_dir / "CODE_ANALYSIS_DONE.yaml"
    with status_path.open("r", encoding="utf-8") as handle:
        status = yaml.safe_load(handle) or {}
    if status.get("status") != "accepted":
        raise RuntimeError("代码结构分析尚未标记为 accepted，不能启动正式实验")

