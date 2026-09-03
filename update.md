# SAE 项目更新说明

## 1. 对比范围

本文档比较以下两个 Git 版本：

| 版本 | Commit | 含义 |
|---|---|---|
| 最初版本 | `9fe95aa6186a77adca9e165666b2082bfba8efe2` | 项目初始结构与五阶段最小实现 |
| 当前版本 | `4486166ebb46abbaf260a67de1f6c3af0f8e787d` | 完善 P02–P05 证据链后的版本 |

Git 差异统计为：41 个文件发生变化，增加约 3,717 行，删除约 344 行。本次更新的重点不是扩大数据集或直接复用论文作者代码，而是修复旧版实验中证据不足、指标偏宽松和因果干预没有实际生效的问题。

当前项目仍属于在统一 CIFAR-100 数据集上的机制复现（`ADAPTED`）。代码已经增强不代表实验结论已经成立；P02–P05 必须重新运行并依据新输出判断。

## 2. 总体变化

最初版本已经建立以下五阶段链路：

```text
P01 superposition
  → P02 SAE feature
  → P03 单个 VLM 单义性
  → P04 两个 VLM 的共享 SAE
  → P05 feature splitting / absorption
```

当前版本在这条链路上增加了四类关键能力：

1. 公平对照：不再只报告 SAE 自身结果，而是与 raw neuron、随机 latent、样本置换和错误配对训练进行比较。
2. 数据隔离：模型或 latent 选择在训练/验证数据完成，最终结论在独立测试集评价。
3. 真正干预：不再只计算静态相关或零值 clamp，而是执行 latent injection、ablation、decoder 重建和输出变化测量。
4. 证据边界：明确区分重建质量、稀疏性、可解释性、相关性和因果性，避免由单一高分直接宣称论文结论成立。

五阶段的主要变化可概括为：

| 阶段 | 最初版本 | 当前版本 |
|---|---|---|
| P01 | 训练 toy superposition 模型 | 修复保存 requires-grad tensor 时调用 NumPy 的错误 |
| P02 | 单一 ReLU+L1 SAE、最佳 latent F1、方向最大匹配 | SAE 扫描、验证集选型、raw neuron 对照、purity、一对一匹配和 toy 因果消融 |
| P03 | CLIP SAE、标签语义 MS、top images、centroid probe clamp | 独立 SigLIP 语义参考、支持度过滤 MS、正激活图片、CLIP zero-shot injection/ablation 与剂量响应 |
| P04 | cross-R²、FE、CFP | 逐 latent 语义对齐、bootstrap、样本/latent 置换和错误图片配对训练对照 |
| P05 | 同一训练数据逐样本寻找最大贡献 latent | discovery/validation/test、修正 splitting、child specificity、FDR、固定候选消融和匹配随机对照 |

## 3. 基础工程修复

### 3.1 CIFAR-100 类别名称修复

修改文件：

```text
src/sae_repro/data/concepts.py
```

最初版本中部分 fine class 使用了复数或非 torchvision 官方名称，例如：

```text
orchids、poppies、bottles、apples、computer_keyboard
```

当前版本改为 CIFAR-100/torchvision 实际名称：

```text
orchid、poppy、bottle、apple、keyboard
```

该修改解决了 `make prepare` 中“存在未知 CIFAR-100 fine 类别”的错误，并保证 fine→coarse 映射覆盖全部 100 个类别。

### 3.2 P01 tensor 保存修复

修改文件：

```text
src/sae_repro/stages/p01_superposition.py
```

最初版本在保存 hidden 时，tensor 仍保留梯度图，导致：

```text
RuntimeError: Can't call numpy() on Tensor that requires grad
```

当前版本在最终特征提取外层增加 `torch.no_grad()`，确保保存的训练/测试 hidden 和 feature directions 都是推理结果，不携带梯度。

### 3.3 本地模型路径

P03/P04 默认模型名称改为项目内离线路径：

```text
models/openai-clip-vit-base-patch32
models/google-siglip-base-patch16-224
```

这使服务器在 Hugging Face 网络不可用时仍可使用已上传的本地模型。

### 3.4 GitHub 大文件边界

`.gitignore` 新增或加强了以下规则：

```text
/models/
/data/raw/
/data/shared/
/outputs/
*.pt
*.npy
*.npz
*.bin
*.safetensors
*.msgpack
*.h5
*.tar.gz
```

因此 GitHub 只保存源码、配置、测试和说明文档，不上传数据集、模型权重、checkpoint 和完整实验输出。

## 4. P02：从 superposition 中恢复可解释 feature

### 4.1 最初版本的不足

最初版本只训练一个 ReLU+L1 SAE。旧结果虽然具有很高的重建 R²，但平均 L0 约为 62/128，latent 并不稀疏。因此高重建不能直接证明 SAE 已经把 superposition 拆成单个 feature。

此外，最初版本还缺少：

- raw neuron 与 SAE latent 的公平比较；
- 训练、验证和测试职责分离；
- 一对一 dictionary recovery；
- latent 概念纯度与覆盖率；
- 对已知 toy feature 的因果消融。

### 4.2 当前版本的改进

修改文件：

```text
configs/p02.yaml
src/sae_repro/stages/p02_interpretable_sae.py
src/sae_repro/metrics/concepts.py
src/sae_repro/metrics/disentanglement.py
```

新增功能：

1. 同时扫描 ReLU+L1 SAE 和 BatchTopK SAE。
2. 扫描多个 L1 系数和 TopK 值，而不是依赖单一超参数。
3. 从训练集内部划分验证集，并按照重建、L0、概念 F1 和方向恢复综合选择模型。
4. raw neuron 与 SAE latent 使用同一组候选阈值、同一验证选择协议和同一独立测试集。
5. 计算 latent 的 fine/coarse purity、entropy、支持数和 coverage。
6. 使用 Hungarian matching 做 ground-truth feature direction 与 SAE decoder 的一对一匹配，禁止多个真实 feature 重复使用同一个 latent。
7. 消融映射到目标概念的 SAE latent，再通过 P01 decoder 计算目标概念与无关概念输出变化。

### 4.3 新增 P02 输出

```text
outputs/p02/sweep_metrics.json
outputs/p02/interpretability_comparison.json
outputs/p02/disentanglement_metrics.json
outputs/p02/toy_interventions.json
```

新的判断标准是：只有同时满足较高重建、较低 L0、优于 raw neuron、较高 purity、较好一对一方向恢复和有方向性的消融效应，才支持“SAE 拆解了 superposition”。

## 5. P03：单个 VLM 的 SAE 单义性与因果干预

### 5.1 最初版本的不足

最初版本已经完成：

```text
CLIP 激活 → SAE 系数 → top images → MS
```

但存在三个主要问题：

1. MS 可能由只有少量正激活样本的 latent 获得虚高分数。
2. top-image 不足时会混入零激活图片，容易被误当作概念证据。
3. 95% 分位数 clamp 可能仍为零，导致干预前后 latent 完全没有变化，输出差异只是数值误差。

### 5.2 当前版本的改进

修改或新增文件：

```text
configs/p03.yaml
src/sae_repro/stages/p03_single_vlm.py
src/sae_repro/metrics/monosemanticity.py
src/sae_repro/metrics/interventions.py
src/sae_repro/models/clip_zero_shot.py
src/sae_repro/visualization/top_images.py
src/sae_repro/visualization/interventions.py
```

新增功能：

1. 默认使用独立 SigLIP 图像 embedding 作为语义参考，避免被解释的 CLIP 自己给自己评分。
2. 保存每个 latent 的正激活支持数；支持数小于阈值的 MS 标记为不可靠。
3. 对高 MS latent 进行重复子采样，输出稳定性区间。
4. top-image 只保留真正大于激活阈值的图片，不足位置使用 `-1`，可视化不再填充零激活样本。
5. clamp 值只在正激活值中取分位数，避免大量零值把干预强度压到零。
6. 同时执行 latent injection 和 ablation。
7. 使用多档 dose，检查输出是否具有方向一致的剂量响应。
8. 使用 firing frequency 匹配的 control latent，避免把一般性扰动误认为目标概念因果效应。
9. 加载 CLIP 文本塔，用多提示词构造 100 个 CIFAR fine class 原型。
10. 将 SAE 标准化空间还原为 CLIP embedding 空间，直接计算真实 CLIP zero-shot logits 和分类准确率。

### 5.3 新增 P03 输出

```text
outputs/p03/ms_latent_supported.npy
outputs/p03/ms_support_counts.npy
outputs/p03/top_positive_counts.npy
outputs/p03/top_positive_image_grids/
outputs/p03/test_semantic_reference_raw.npy
outputs/p03/test_semantic_reference_sample_ids.npy
outputs/p03/clip_text_prototypes.npy
outputs/p03/intervention_results.json
outputs/p03/intervention_plots/dose_response.png
outputs/p03/intervention_plots/ablation_effects.png
```

当前 P03 的因果结论仍限定在 CLIP 最终图像 embedding 和 zero-shot logits，不等同于原论文的中间 patch/token 干预或 LLaVA 文本生成干预。

## 6. P04：两个 VLM 的 Universal SAE

### 6.1 最初版本的不足

最初版本主要报告：

- 2×2 cross-reconstruction R²；
- firing entropy；
- co-fire proportion；
- 两侧 decoder contribution energy。

这些结果说明共享 code 可用于跨模型重建，但 Universal SAE 的训练目标本身就强制两个模型使用同一 latent 索引，因此不能只凭同索引、FE 或 CFP 宣称两个模型发现了相同语义 feature。

### 6.2 当前版本的改进

修改或新增文件：

```text
configs/p04.yaml
src/sae_repro/stages/p04_universal_sae.py
src/sae_repro/metrics/universality.py
src/sae_repro/metrics/cross_model_alignment.py
src/sae_repro/visualization/cross_model.py
```

对每个 CLIP/SigLIP 同索引 latent 新增：

- 两侧正激活支持数；
- 激活 Pearson 相关；
- firing Jaccard；
- top-image Jaccard；
- fine/coarse 主标签和标签一致率；
- fine/coarse 标签纯度；
- fine/coarse 标签分布余弦相似度；
- minimum-support coverage。

新增统计与负对照：

1. 以 latent 为抽样单位计算 bootstrap 置信区间。
2. 打乱测试样本对应关系，建立激活相关和 co-firing 零假设。
3. 打乱 latent 索引对应关系，建立标签语义一致性零假设。
4. 计算目标样本被打乱后的 cross-reconstruction R²。
5. 使用相同模型结构、相同 K、相同训练步数，但打乱 CLIP–SigLIP 训练图片配对，额外训练一个错误配对 Universal SAE。
6. 在正确配对测试集上比较正常模型和错误配对模型的逐 latent 对齐结果。

### 6.3 新增 P04 输出

```text
outputs/p04/latent_alignment.json
outputs/p04/latent_alignment.csv
outputs/p04/universality_controls.json
outputs/p04/shuffled_pair_train_permutation.npy
outputs/p04/shuffled_target_test_permutation.npy
outputs/p04/shuffled_pair_latent_alignment.json
outputs/p04/alignment_plots/latent_alignment_scatter.png
outputs/p04/alignment_plots/alignment_null_comparison.png
```

当 `save_control_checkpoint: true` 时，还会保存：

```text
outputs/p04/universal_sae_shuffled_control.pt
```

默认不保存该对照 checkpoint，以减少磁盘占用。

P04 现在默认额外训练一个同规模错误配对对照，因此 Universal SAE 部分的运行时间大约是最初版本的两倍。SigLIP 原始激活缓存仍会复用，不会因为该对照再次提取图片激活。

## 7. P05：feature splitting 与 feature absorption

### 7.1 最初版本的不足

最初版本存在以下偏差来源：

1. 第一个 main latent 从 F1=0 开始的提升也被计为一次 splitting。
2. main latent、候选 latent 和最终评价使用同一批训练数据。
3. 对每个 false-negative 样本单独寻找贡献最大的 latent，容易系统性高估 absorption。
4. 候选只需要沿 parent probe 方向有正贡献，不要求对应具体 fine child。
5. 只计算线性 decoder contribution，没有显式构造 latent 置零后的重建。
6. 没有标签置换、随机 latent 或匹配属性的零假设。
7. absorption rate 只使用 parent positives 作为分母，不能直接回答 recall holes 中有多少被接管。

### 7.2 当前版本的改进

修改或新增文件：

```text
configs/p05.yaml
src/sae_repro/stages/p05_absorption.py
src/sae_repro/metrics/absorption.py
src/sae_repro/metrics/absorption_controls.py
src/sae_repro/metrics/splitting.py
src/sae_repro/visualization/absorption.py
```

新的实验协议：

```text
train
  ├─ discovery：选择 main latent、split 顺序、parent direction、absorption candidates
  └─ validation：固定规则诊断

test：固定所有规则并报告最终结果
```

具体改进：

1. 按 coarse class 分层划分 discovery 和 validation。
2. splitting 顺序只在 discovery 上确定。
3. 第一个 main latent 仅作为 baseline，不再计入 additional splitting。
4. absorption candidate 必须在 discovery 中覆盖 main latent 的 false negatives。
5. 候选必须具有足够的 parent decoder projection 和实际贡献。
6. 候选必须在 parent 内偏向某个 fine child，并报告 child purity、lift 和 support。
7. fine-label specificity 使用标签置换检验，并在同一 parent 内执行 Benjamini–Hochberg FDR 校正。
8. 每个 fine child 最多选择一个最佳候选，避免多个相似 latent 重复堆高消融效应。
9. validation/test 阶段禁止重新选择候选。
10. 把固定候选 latent 同时置零，经 decoder 重新构造表示，再计算 parent probe 分数下降。
11. 随机对照同时匹配 firing rate、decoder norm 和 parent projection。
12. 报告匹配随机对照的均值、标准差和经验 p 值。
13. 同时报告：
    - `absorption_rate_over_tested_false_negatives`；
    - `absorption_rate_over_parent_positives`；
    - `matching_child_rate_over_absorbed`；
    - `parent_probe_score_correlation`。

### 7.3 新增 P05 输出

```text
outputs/p05/discovery_indices.npy
outputs/p05/validation_indices.npy
outputs/p05/controlled_p02.json
outputs/p05/universal_clip_p04.json
outputs/p05/candidate_latents.csv
outputs/p05/plots/controlled_p02_absorption_vs_null.png
outputs/p05/plots/controlled_p02_splitting_curves.png
outputs/p05/plots/universal_clip_p04_absorption_vs_null.png
outputs/p05/plots/universal_clip_p04_splitting_curves.png
```

P05 的因果结论仍是“SAE decoder 重建空间内，固定 child-specific latent 对线性 parent probe 的因果贡献”，不是 VLM 最终分类或文本生成行为的完整因果证明。

## 8. 新增源码模块

| 文件 | 作用 |
|---|---|
| `metrics/disentanglement.py` | raw/SAE 公平映射、latent purity/entropy、Hungarian 一对一方向匹配和 P01 toy 消融 |
| `metrics/interventions.py` | P03 正激活分位数、injection、ablation、剂量响应和 matched control |
| `models/clip_zero_shot.py` | CLIP 文本原型构造、标准化 embedding 还原和 zero-shot logits |
| `metrics/cross_model_alignment.py` | P04 逐 latent 激活/语义对齐、置换检验和 bootstrap |
| `metrics/splitting.py` | discovery 固定 latent 顺序、留出集 splitting 曲线和首 latent 基线修正 |
| `metrics/absorption_controls.py` | child specificity、FDR、decoder 重建消融和 matched-random control |
| `visualization/interventions.py` | P03 剂量响应和消融效果图 |
| `visualization/cross_model.py` | P04 跨模型对齐散点图和置换对照图 |
| `visualization/absorption.py` | P05 absorption/null 柱状图和 splitting 曲线 |

## 9. 测试增强

新增测试文件：

```text
tests/test_disentanglement.py
tests/test_interventions.py
tests/test_cross_model_alignment.py
tests/test_splitting.py
```

更新测试文件：

```text
tests/test_concepts.py
tests/test_ms.py
tests/test_absorption.py
```

新测试覆盖：

- top-image 不得用零激活样本补充；
- MS 支持度过滤；
- raw/SAE 阈值必须从验证集选择；
- Hungarian matching 不得重复使用 latent；
- 稀疏 latent clamp 必须忽略零值；
- CLIP 标准化空间必须正确还原；
- 相同跨模型 latent 应获得高对齐指标；
- 打乱目标样本应降低 cross-reconstruction；
- 第一个 latent 不得被计为额外 splitting；
- 固定 child candidate 消融应产生可测 parent effect。

本次代码检查完成了：

```text
python -m compileall -q src experiments scripts tests
```

并在具有 PyTorch 的本地环境中手动执行了 16 个无 fixture 测试函数。由于当前 Mac 基础环境没有安装 pytest，正式服务器环境仍应执行：

```bash
make check
```

## 10. 文档与代码分析更新

以下代码结构材料已同步到当前实现：

```text
docs/code_structure/CODE_ANALYSIS_DONE.yaml
docs/code_structure/assumptions.yaml
docs/code_structure/call_graph.mmd
docs/code_structure/equation_code_map.csv
docs/code_structure/module_inventory.csv
docs/code_structure/repository_tree.md
docs/code_structure/tensor_shapes.md
README.md
目录.md
```

主要补充了：

- P02–P05 的新模块和调用关系；
- 论文概念—代码模块—输出文件—测试之间的映射；
- discovery/validation/test 张量边界；
- 正确配对、置换和错误配对训练对照；
- SAE 表示因果与 VLM 行为因果之间的边界。

## 11. 依赖变化

`requirements.txt` 相比最初提交没有新增依赖。本次新增实现使用的 PyTorch、NumPy、SciPy、pandas、matplotlib 和 pytest 已经包含在原 requirements 中。

模型和数据仍需单独准备，不进入 Git：

```text
data/raw/cifar-100-python.tar.gz
models/openai-clip-vit-base-patch32/
models/google-siglip-base-patch16-224/
```

## 12. 重新运行要求

最初版本生成的 `outputs/p02` 至 `outputs/p05` 不包含新增指标，不能直接作为当前代码的正式结果。建议在服务器按以下顺序重新运行：

```bash
cd /opt/data/private/yzx/SAE
conda activate SAE

make analyze
make check
make p02
make p03
make p04
make p05
```

如果 P02/P03 已经由当前 commit `4486166` 重新运行，可以只执行：

```bash
make p04
make p05
```

每次报告结果时应同时检查：

```text
outputs/pXX/manifest.json
outputs/pXX/metrics.json
```

不得用旧输出解释新代码，也不得因为增加了指标代码就预先宣称论文结论已经复现。

## 13. 更新后的证据链

当前代码希望验证的完整链路为：

```text
已知稀疏概念
  ↓
低维 superposition 表示
  ↓
SAE 稀疏系数与 feature 拆解对照
  ↓
真实 CLIP 图像激活的 SAE 表示
  ↓
正激活概念可视化与支持度过滤单义性
  ↓
CLIP zero-shot injection/ablation 因果评价
  ↓
CLIP–SigLIP 共享 latent 与多类负对照
  ↓
固定 child-specific latent 的 absorption 消融
```

相较最初版本，当前项目不再把“高重建”“共享索引”“高 MS”或“正 decoder contribution”单独视为结论成立，而是要求这些现象同时通过稀疏性、独立测试集、语义一致性、因果干预和零假设对照。
