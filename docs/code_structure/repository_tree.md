# 初始代码结构分析

## 1. 入口与依赖顺序

`src/sae_repro/cli.py` 是统一入口，调用 `stages/pipeline.py`。流水线顺序固定为：

```text
prepare_shared_cifar100
→ p01_superposition.run
→ p02_interpretable_sae.run
→ p03_single_vlm.run
→ p04_universal_sae.run
→ p05_absorption.run
```

每个阶段先通过 `core/preflight.py` 检查本目录的代码分析文件，再检查上游 `manifest.json` 和数组文件。阶段不会自动用随机数组替代缺失产物。

## 2. 主要模块

- `core/`：合并 YAML、解析项目路径、选择 MPS/CUDA/CPU、固定随机种子、写 manifest、执行分析门禁。
- `data/`：定义 CIFAR-100 coarse→fine 层级，构造 120 维概念矩阵，固定五阶段共享索引。
- `models/vision.py`：顺序加载 CLIP 或 SigLIP 视觉塔，按同一索引缓存图像级激活。
- `sae/models.py`：基础 ReLU SAE、BatchTopK SAE、Matryoshka BatchTopK SAE 和双模型 Universal SAE。
- `sae/trainer.py`：单 SAE 与 Universal SAE 的小批量训练、校准、编码和重建。
- `metrics/`：重建、稀疏、MS、concept F1、双模型 universality 和 absorption。
- `stages/`：只编排数据、模型、指标和产物，不重复数学实现。
- `experiments/`：每篇论文的薄启动脚本。

## 3. 阶段产物连接

- P01 保存 `train_hidden.npy`，P02 必须直接读取。
- P02 保存 `sae.pt`、`train_latents.npy` 和 `decoder_directions.npy`；P03 复用 P02 实际 expansion factor 并记录概念恢复分数，P05 直接分析这三类表示。
- P03 保存同一索引的 CLIP 激活、SAE latent、MS 和 decoder；P04 直接复用 CLIP 激活，不重复提取。
- P04 只新增 SigLIP 激活，并保存两模型共享 latent 和两个 decoder；P05 分析 CLIP 分支的 parent/child coverage。
- P05 同时输出 P02 受控设置和 P04 真实双 VLM 设置，避免只用单一来源解释 absorption。

## 4. 当前边界

- 当前 VLM 激活是图像级 pooled/projected embedding，不是主体论文的多层 CLS/patch token 全矩阵。
- 当前 P03 因果评价是 class centroid probe，不是 LLaVA 文本生成。
- 当前 P04 只实现两个模型，不实现三模型论文原规模。
- 当前 P05 用 decoder contribution 对 parent probe 的消融效应，不把它表述成生成行为因果证明。
