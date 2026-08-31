from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """显示当前源码树和运行前代码分析文件，供人工检查。"""
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "sae_repro"
    print("项目根目录：", root)
    print("\n核心源码：")
    for path in sorted(source.rglob("*.py")):
        print("-", path.relative_to(root))
    print("\n代码分析文档：")
    analysis = root / "docs" / "code_structure"
    for path in sorted(analysis.iterdir()):
        if path.is_file():
            print("-", path.relative_to(root))
    print("\n请先阅读 repository_tree.md、module_inventory.csv、equation_code_map.csv、tensor_shapes.md。")


if __name__ == "__main__":
    sys.exit(main())

