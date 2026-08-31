# 数据目录

运行 `make prepare` 后会生成：

```text
data/
├── raw/                         # torchvision 下载的 CIFAR-100
└── shared/
    ├── manifest.json            # 类别、随机种子、样本索引和概念定义
    ├── train_indices.npy
    ├── test_indices.npy
    ├── train_fine.npy
    ├── test_fine.npy
    ├── train_coarse.npy
    ├── test_coarse.npy
    ├── train_concepts.npy       # [N_train, 120]
    └── test_concepts.npy        # [N_test, 120]
```

五个阶段只能使用这份 manifest 中的样本索引，确保前后产物严格对应。

