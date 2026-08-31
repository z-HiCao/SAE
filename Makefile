PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src

.PHONY: analyze check prepare p01 p02 p03 p04 p05 all

# 显示当前代码结构分析信息
analyze:
	$(PYTHON) scripts/analyze_structure.py

# 只执行语法和单元测试，不下载模型或数据
check:
	$(PYTHON) -m compileall -q src experiments scripts tests
	$(PYTHON) -m pytest -q

# 准备共享 CIFAR-100 索引和层级概念矩阵
prepare:
	$(PYTHON) -m sae_repro.cli prepare

# 五个阶段按顺序运行，后一阶段会检查前一阶段产物
p01:
	$(PYTHON) -m sae_repro.cli p01

p02:
	$(PYTHON) -m sae_repro.cli p02

p03:
	$(PYTHON) -m sae_repro.cli p03

p04:
	$(PYTHON) -m sae_repro.cli p04

p05:
	$(PYTHON) -m sae_repro.cli p05

all:
	$(PYTHON) -m sae_repro.cli all

