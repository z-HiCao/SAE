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
- `models/clip_zero_shot.py`：构造 CIFAR-100 文本原型，并把 SAE 重建激活转换成 CLIP 图文 logits。
- `sae/models.py`：基础 ReLU SAE、BatchTopK SAE、Matryoshka BatchTopK SAE 和双模型 Universal SAE。
- `sae/trainer.py`：单 SAE 与 Universal SAE 的小批量训练、校准、编码和重建。
- `metrics/`：重建、稀疏、MS、concept F1、一对一分解、因果干预、双模型语义对齐、splitting 和 absorption 对照。
- `visualization/`：绘制正激活概念网格、CLIP 干预、跨模型对齐和 absorption/零假设对比图。
- `stages/`：只编排数据、模型、指标和产物，不重复数学实现。
- `experiments/`：每篇论文的薄启动脚本。

## 3. 阶段产物连接

- P01 保存 `train_hidden.npy`，P02 必须直接读取。
- P02 扫描 ReLU+L1 与 BatchTopK，使用训练集内部验证划分选模型；保存 raw-neuron 对照、latent purity、一对一方向匹配和 toy 消融。P03 只复用所选 P02 模型的 expansion factor，P05 直接分析其 latent 与 decoder。
- P03 保存同一索引的 CLIP 激活、SAE latent、独立 SigLIP 语义参考、支持度过滤 MS 和 CLIP logits 干预；P04 只复用 CLIP 激活，不复用 P03 SAE。
- P04 新增 SigLIP 激活，训练两模型共享 latent 和两个 decoder；逐 latent 评价测试样本及标签语义对齐，并加入样本置换、latent 置换和错误图片配对训练对照。
- P05 同时分析 P02 受控设置和 P04 双 VLM 设置；main/split/candidate 只在 discovery 选择，在 validation/test 使用固定规则执行 decoder 重建消融和匹配随机对照。

## 4. 当前边界

- 当前 VLM 激活是图像级 pooled/projected embedding，不是主体论文的多层 CLS/patch token 全矩阵。
- 当前 P03 因果评价已使用 CLIP 文本塔的 zero-shot logits，并包含正激活分位数、注入、消融、剂量响应和频率匹配对照；它仍不是论文中的 CLIP 中间 token/LLaVA 文本生成干预。
- 当前 P04 只实现两个模型，不实现三模型论文原规模。
- P04 错误图片配对训练对照与主实验结构、训练步数一致，因此默认会让 P04 的 SAE 训练时间约增加一倍。
- 当前 P05 在 SAE decoder 重建空间消融固定 latent 集合，并测量 parent probe 变化；不把它表述成 VLM 生成行为因果证明。
