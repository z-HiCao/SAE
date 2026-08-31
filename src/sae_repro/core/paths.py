from pathlib import Path


def project_root() -> Path:
    """返回 SAE 子项目根目录。"""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """把配置中的相对路径解析为相对于项目根目录的绝对路径。"""
    value = Path(path)
    if value.is_absolute():
        return value
    return project_root() / value

