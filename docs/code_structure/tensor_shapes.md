# 张量形状

| 边界 | 形状 | 含义 |
|---|---:|---|
| CIFAR 图像 | PIL，处理后 `[B,3,H,W]` | 五阶段共享图像 |
| fine labels | `[N]` | 0–99 |
| coarse labels | `[N]` | 0–19 |
| concept matrix | `[N,120]` | fine one-hot 拼接 coarse one-hot |
| P01 bottleneck | `[N,32]` | 已知 feature 的低维叠加激活 |
| P01 feature directions | `[120,32]` | 每个真实 feature 在瓶颈中的方向 |
| P02 latent | `[N,128]` | P01 hidden 的 4× SAE 系数 |
| CLIP activation | `[N,d_clip]` | 图像级 projected embedding 标准化结果 |
| P03 latent | `[N,4×d_clip]` | Matryoshka BatchTopK 系数 |
| semantic reference | `[N,120]` | 用标签层级定义的语义相似空间 |
| MS | `[4×d_clip]` | 每个 latent 一个分数 |
| SigLIP activation | `[N,d_siglip]` | 图像级 pooled embedding 标准化结果 |
| Universal latent | `[N,512]` | CLIP/SigLIP 共享索引空间 |
| cross reconstruction | `[2,2]` | source encoder × target decoder 的 R² |
| FE | `[512]` | 每个共享 latent 的 firing entropy |
| CFP | `[2,512]` | 每模型的共同激活占比 |

