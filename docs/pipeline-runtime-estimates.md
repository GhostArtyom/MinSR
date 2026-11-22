# MinSR 单卡全流程耗时估算

适用于 10×10 周期方格上的 $J_1=1$、$J_2=0.5$ 模型。总耗时包括普通训练、对称性训练和测量。三个阶段使用相同的 `NSAMPLES`，测量的 `STEPS` 固定为 100，包含一次初始热化。

## 全流程耗时估算

训练步数按「普通训练 + 对称性训练」列出。RTX 5060 Ti 和 RTX 4070 Ti 的 1000 样本、2000 + 1000 步数据来自完整运行，其余数值为估算。[复现归档](reproduction-results.md#已完成的复现) · [单步与单轮耗时](implementation-improvements.md#jax-与-pytorch-耗时对比)

| 样本数 | 训练步数 | 实现 | RTX 5060 Ti | RTX 4070 Ti | RTX 4090 | RTX 5090 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | 2000 + 1000 | JAX 原版 | 12.4 小时 | 7.5 小时 | 3.7 小时 | 2.9 小时 |
| 1000 | 2000 + 1000 | PyTorch 优化版 | 3.8 小时 | 3.1 小时 | 1.5 小时 | 1.2 小时 |
| 1000 | 20000 + 10000 | JAX 原版 | 101.0 小时 (4.2 天) | 61.5 小时 (2.6 天) | 29.9 小时 (1.2 天) | 23.6 小时 |
| 1000 | 20000 + 10000 | PyTorch 优化版 | 32.7 小时 (1.4 天) | 27.6 小时 (1.2 天) | 13.5 小时 | 10.6 小时 |

## 分阶段耗时估算

### RTX 5060 Ti (16 GB)

| 样本数 | 训练步数 | 实现 | 普通训练 | 对称性训练 | 测量 | 总耗时 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | 2000 + 1000 | JAX 原版 | 1.4 小时 | 8.5 小时 | 2.5 小时 | 12.4 小时 |
| 1000 | 2000 + 1000 | PyTorch 优化版 | 35 分钟 | 2.6 小时 | 38 分钟 | 3.8 小时 |
| 1000 | 20000 + 10000 | JAX 原版 | 14.1 小时 | 84.4 小时 (3.5 天) | 2.5 小时 | 101.0 小时 (4.2 天) |
| 1000 | 20000 + 10000 | PyTorch 优化版 | 5.7 小时 | 26.3 小时 (1.1 天) | 38 分钟 | 32.7 小时 (1.4 天) |

### RTX 4070 Ti (12 GB)

| 样本数 | 训练步数 | 实现 | 普通训练 | 对称性训练 | 测量 | 总耗时 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | 2000 + 1000 | JAX 原版 | 51 分钟 | 5.2 小时 | 1.5 小时 | 7.5 小时 |
| 1000 | 2000 + 1000 | PyTorch 优化版 | 27 分钟 | 2.3 小时 | 25 分钟 | 3.1 小时 |
| 1000 | 20000 + 10000 | JAX 原版 | 8.5 小时 | 51.5 小时 (2.1 天) | 1.5 小时 | 61.5 小时 (2.6 天) |
| 1000 | 20000 + 10000 | PyTorch 优化版 | 4.5 小时 | 22.7 小时 | 25 分钟 | 27.6 小时 (1.2 天) |

## 显卡理论峰值算力

| 显卡 | 显存 | FP32 (TFLOPS) | FP64 (TFLOPS) |
|---|---:|---:|---:|
| [RTX 5060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/) | 16 GB | 23.7 | 0.37 |
| [RTX 4070 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/) | 12 GB | 40.1 | 0.63 |
| [RTX 4090](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) | 24 GB | 82.6 | 1.29 |
| [RTX 5090](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) | 32 GB | 104.8 | 1.64 |


## 估算依据

两种实现的基准均为 16 层、16 通道、3×3 卷积和 1000 样本。普通训练取第 2000 步的累计耗时，对称性训练取第 1000 步的累计耗时，测量取 100 轮摘要的 `seconds` 字段。各阶段的实测耗时含热化，训练耗时还包含绘图和存盘。总耗时先加总再舍入。

| 实现 | 设备 | 普通训练日志 | 对称性训练日志 | 测量摘要 |
| --- | --- | --- | --- | --- |
| JAX 原版 | RTX 5060 Ti | [`training.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_030050/training.log) | [`training_symm.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_051106/training_symm.log) | [`summary.json`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/summary.json) |
| JAX 原版 | RTX 4070 Ti | [`training.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_054229/training.log) | [`training_symm.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_063326/training_symm.log) | [`summary.json`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261120_022539/summary.json) |
| PyTorch 优化版 | RTX 5060 Ti | [`training.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_153201/training.log) | [`training_symm.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_160638/training_symm.log) | [`summary.json`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_184508/summary.json) |
| PyTorch 优化版 | RTX 4070 Ti | [`training.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_123654/training.log) | [`training_symm.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_130358/training_symm.log) | [`summary.json`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_152043/summary.json) |

### JAX 原版

延长训练时，RTX 5060 Ti 按两份训练日志首步后的平均耗时外推，初始化只计一次。RTX 4070 Ti 按训练步数比例放大两段实测耗时。测量保持 100 轮，沿用实测耗时。

### PyTorch 优化版

延长训练时，两张卡都用各自日志中第 10 步后的平均耗时乘以新增训练步数，再加上对应阶段的实测耗时。测量保持 100 轮，沿用实测耗时。

### RTX 4090 / 5090

两种实现均以各自的 RTX 4070 Ti 同配置全流程耗时为基准。根据上方算力表，分别计算目标卡的 FP32、FP64 峰值相对 RTX 4070 Ti 的倍率并取平均，RTX 4090 约为 2.05 倍，RTX 5090 约为 2.61 倍。以基准耗时除以未舍入的倍率得到估算值。
