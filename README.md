# 五篇 SAE 论文的连续最小复现

本目录把五篇论文重组为一条使用同一数据集、共享样本索引和连续实验产物的复现链。当前提交是**初始项目结构与初始代码**，未安装依赖、未下载数据/模型、未运行训练，因此没有任何结果可以标记为“已复现”。

## 1. 为什么选择 CIFAR-100

CIFAR-100 有 50,000 张训练图像、10,000 张测试图像、100 个 fine classes 和 20 个 coarse superclasses。每个 coarse class 恰好包含 5 个 fine classes，天然提供：

- 同一批图像供五个阶段复用；
- 稀疏的 120 维概念向量：100 个 fine 概念 + 20 个 coarse 概念，每张图只激活 2 维；
- `coarse → fine` 的已知层级，可在第五阶段检查 feature splitting 和 feature absorption；
- 数据规模适合 Apple Silicon 上做最小实验。

这是一项**机制链路复现/适配**：第一、二、四、五篇论文的原始实验对象并不全是 CIFAR-100。项目保留论文的核心干预逻辑，但不会把 CIFAR-100 适配结果写成论文原数值的严格复现。第三篇主体论文仍使用真实 VLM 激活、SAE、单义性评价和 latent 干预链路。

## 2. 五阶段产物链

```text
CIFAR-100 图像 + fine/coarse 层级标签
│
├─ P01 Toy Models of Superposition
│    输入：120 维稀疏概念向量
│    输出：32 维叠加瓶颈激活 H、feature directions、叠加指标
│
├─ P02 Sparse Autoencoders Find Highly Interpretable Features
│    输入：P01 的 H 与已知 120 维概念标签
│    输出：SAE checkpoint、稀疏系数 Z、概念恢复 F1、字典方向匹配
│
├─ P03 Monosemantic Features in a Single VLM
│    输入：同一图像索引的 CLIP 激活 + P02 验证过的 SAE 组件
│    输出：CLIP SAE、latent top images、MS、重建/L0、probe 干预效应
│
├─ P04 Universal SAE on Two VLMs
│    输入：P03 的 CLIP 激活 + 同图像的 SigLIP 激活
│    输出：共享 latent、双向 cross-reconstruction、FE/CFP universality
│
└─ P05 Feature Splitting and Absorption
     输入：P02/P04 latent、decoder、CIFAR coarse→fine 层级
     输出：parent latent 的 recall hole、splitting 数、absorption 候选与消融效应
```

所有阶段通过 `sample_id`、固定的 train/test indices 和同一份 `concepts.npy` 对齐。任何样本数或顺序不一致都会触发错误，而不是静默继续。

## 3. 当前实现范围

### P01：证明 superposition 可出现

- 用 CIFAR-100 的 120 维层级概念向量训练 `m<n` 的 tied-weight 重建模型。
- 比较线性输出与 ReLU 输出。
- 对零概念位注入不同密度的受控特征，形成 sparsity sweep。
- 输出被表示特征数、Gram interference、重建误差和是否满足“表示特征数大于瓶颈维数”。
- 原始稀疏设置的瓶颈激活直接成为 P02 输入。

### P02：用 SAE 从已知叠加中恢复 feature

- 在 P01 瓶颈激活上训练 ReLU + L1 SAE。
- 用已知 CIFAR 概念标签评价每个 concept 的最佳 latent F1。
- 将 SAE decoder directions 与 P01 ground-truth feature directions 做 cosine matching。
- 输出 `Z` 和 checkpoint，作为真实 VLM SAE 前的实现验证。

### P03：单个 VLM 的 SAE 单义链路

- 使用 `openai/clip-vit-base-patch32` 提取同一 CIFAR-100 样本的 image embeddings。
- 训练 Matryoshka BatchTopK SAE。
- 计算 R²、L0、dead latent、top activating image indices。
- 使用 CIFAR fine/coarse concept embeddings 作为独立的语义参考计算 MS；该设置避免直接用被解释的 CLIP 自己给自己评分，但它只评价标签语义。
- 对 SAE latent 做正向钳制，经 decoder 重建后观察目标 class centroid probe 的 logit 变化，作为本地可运行的因果代理。

### P04：两个 VLM 的共享 SAE

- 复用 P03 缓存的 CLIP 激活。
- 顺序加载 `google/siglip-base-patch16-224`，提取完全相同样本的 SigLIP 激活，避免同时驻留两个模型。
- 训练两个 encoder、两个 decoder、一个共享索引空间的 Universal SAE。
- 输出 2×2 cross-reconstruction R²、firing entropy（FE）、co-fire proportion（CFP）和两模型 latent 激活。

### P05：在同一视觉层级上检查 absorption

- 对 20 个 coarse parent concepts 找到最能分类它的主 latent。
- 用 k 个 latent 的并集 F1 增量检查 splitting。
- 在 parent 为真但主 latent 不激活的样本中，搜索仍携带 parent probe 方向且有正向消融效应的 child-specific latent。
- 同时分析 P02 的受控叠加 SAE 和 P04 的双模型共享 SAE。
- 当前指标是保守的线性 probe/decoder 因果代理；后续可扩展为对 VLM 真实输出或 LLaVA 生成的干预。

## 4. 目录

```text
SAE/
├── README.md
├── requirements.txt
├── Makefile
├── configs/
│   ├── base.yaml
│   └── p01.yaml ... p05.yaml
├── src/sae_repro/
│   ├── core/          # 配置、设备、路径、产物和运行门禁
│   ├── data/          # CIFAR-100 层级、共享索引和概念矩阵
│   ├── models/        # CLIP/SigLIP 顺序激活提取
│   ├── sae/           # ReLU、BatchTopK、Matryoshka、Universal SAE
│   ├── metrics/       # R²、L0、MS、概念 F1、universality、absorption
│   └── stages/        # P01 至 P05 的连续实验入口
├── experiments/       # 每篇论文的薄启动脚本
├── docs/code_structure/
├── scripts/
├── tests/
├── data/              # 下载后生成；不会提交大文件
└── outputs/           # 运行后生成；每阶段有 manifest 和 metrics
```

## 5. 环境

建议使用 Python 3.11 或 3.12 新建独立环境。当前机器的系统 Python 3.13 不一定与所有科学计算 wheel 组合有同样充分的测试覆盖。

```bash
cd /Users/cz/Desktop/AE相关概念学习论文/SAE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

本任务要求本次不安装依赖，因此以上命令只作为后续使用说明。

## 6. 本地 Mac 能否训练

当前机器是 Apple M4、16GB 统一内存。PyTorch 的 MPS backend 可以在 Apple Silicon 上使用 GPU 加速，本项目会按 `mps → cuda → cpu` 自动选择设备。

| 阶段 | M4 16GB 本地可行性 | 建议 |
|---|---|---|
| P01 | 可行 | 可直接跑 50k 概念向量；先用默认 minimal 配置 |
| P02 | 可行 | 32 维输入、小 SAE，训练压力很低 |
| P03 | 可行但需控制规模 | 先缓存 5k train / 1k test 的 CLIP-Base 激活，模型推理 batch 8–16，SAE batch 256 |
| P04 | 可做最小实验 | CLIP/SigLIP 必须顺序加载；只训练缓存激活上的小 USAE；不要同时把两模型放入内存 |
| P05 | 可行 | 主要是 latent/probe 统计和小规模消融 |
| 论文原规模 | 不可取 | ImageNet 全量、ViT-L、SoViT-400m、64× latent、10^5 steps 和 LLaVA-7B 建议使用 24–80GB NVIDIA GPU |

若某些算子不支持 MPS，可在运行前设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，但回退 CPU 会变慢。为降低内存风险，初始代码只缓存 float32 激活，模型提取后立刻移回 CPU 并释放。

## 7. 使用顺序

代码结构分析文件已经随初始骨架建立。正式运行前先阅读：

```bash
make analyze
```

依赖安装后，先运行测试，再按顺序执行：

```bash
make check
make prepare
make p01
make p02
make p03
make p04
make p05
```

或使用：

```bash
python -m sae_repro.cli all
```

`all` 不会跳过缺失的上游产物。P03/P04 首次运行需要联网下载 CIFAR-100 和 Hugging Face 模型权重。

## 8. 结果标签

- `PAPER_REPORTED`：论文原文中的结果。
- `REPRODUCED`：当前项目真实运行产生且带 run manifest 的结果。
- `ADAPTED`：在统一 CIFAR-100 设置上的机制复现。
- `OBSERVED`：单次定性观察。
- `NOT_RUN`：尚未运行。

当前所有实验状态均为 `NOT_RUN`。P01、P02、P04、P05 的 CIFAR-100 结果应标为 `ADAPTED`；P03 只有在明确对齐模型、层、数据和评价协议后，才能讨论接近论文复现，默认仍是缩小规模的 `ADAPTED`。

