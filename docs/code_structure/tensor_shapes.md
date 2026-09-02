# 张量形状

| 边界 | 形状 | 含义 |
|---|---:|---|
| CIFAR 图像 | PIL，处理后 `[B,3,H,W]` | 五阶段共享图像 |
| fine labels | `[N]` | 0–99 |
| coarse labels | `[N]` | 0–19 |
| concept matrix | `[N,120]` | fine one-hot 拼接 coarse one-hot |
| P01 bottleneck | `[N,32]` | 已知 feature 的低维叠加激活 |
| P01 feature directions | `[120,32]` | 每个真实 feature 在瓶颈中的方向 |
| P02 latent | `[N,expansion×32]` | 验证集从 ReLU+L1/BatchTopK 候选中选择的 SAE 系数；默认扫描 4× |
| CLIP activation | `[N,d_clip]` | 图像级 projected embedding 标准化结果 |
| P03 latent | `[N,4×d_clip]` | Matryoshka BatchTopK 系数 |
| semantic reference | `[N,d_semantic]` | 默认独立 SigLIP 图像嵌入；可配置为 `[N,120]` 标签后备方案 |
| MS | `[4×d_clip]` | 每个 latent 一个分数 |
| MS support | `[4×d_clip]` | 每个 latent 的测试集正激活样本数 |
| CLIP text prototypes | `[100,d_clip]` | 100 个 fine class 的提示词集平均文本方向 |
| SigLIP activation | `[N,d_siglip]` | 图像级 pooled embedding 标准化结果 |
| Universal latent | `[N,512]` | CLIP/SigLIP 共享索引空间 |
| cross reconstruction | `[2,2]` | source encoder × target decoder 的 R² |
| FE | `[512]` | 每个共享 latent 的 firing entropy |
| CFP | `[2,512]` | 每模型的共同激活占比 |
| P04 latent alignment rows | 512 行 | 同索引 latent 的支持数、相关、Jaccard、标签纯度与语义分布相似度 |
| P04 shuffled pair permutation | `[N_train]` | 错配训练对照使用的 SigLIP 样本排列 |
| P05 discovery/validation index | `[N_fit]` / `[N_val]` | 按 coarse 分层得到，规则只在 fit 上选择 |
| P05 candidate table | 每个通过基础筛查的 latent 一行 | parent/child、coverage、decoder effect、purity/lift、p/q 和 selected |
