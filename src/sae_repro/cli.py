from __future__ import annotations

import argparse

from sae_repro.stages.pipeline import run_all, run_prepare, run_stage


def main() -> None:
    """提供 prepare、单阶段和全链路命令。"""
    parser = argparse.ArgumentParser(description="五篇 SAE 论文的连续 CIFAR-100 复现")
    parser.add_argument("command", choices=["prepare", "p01", "p02", "p03", "p04", "p05", "all"])
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        run_prepare()
    elif arguments.command == "all":
        run_all()
    else:
        run_stage(arguments.command)


if __name__ == "__main__":
    main()

