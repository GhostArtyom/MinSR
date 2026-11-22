# MinSR 实现改进

## 数值一致性验证

PyTorch 优化沿用原版的 Heisenberg 算符、Metropolis 采样和 MinSR 更新公式。固定权重与构型的数值对照已通过。浮点运算顺序仍可能影响采样的接受判定，长期训练的能量和方差按统一容差验收。[完整验收结果](pytorch-migration-plan.md#完整验收结果)

## JAX 与 PyTorch 耗时对比

下表使用 10×10 方格、16 层、16 通道、3×3 卷积和 1000 个样本的完整运行。单步或单轮耗时取第 11 步或第 11 轮起的中位数，训练另排除绘图和存盘的步。总耗时为三阶段的实测耗时之和，含热化、训练绘图和存盘。JAX 基线来自启用设备端局域能量前的归档，PyTorch 版本为 `2.10.0+cu128`。[各阶段日志](reproduction-results.md#已完成的复现) · [各阶段实测与全流程估算](pipeline-runtime-estimates.md)

| 设备 | 实现 | 普通训练（秒/步） | 对称性训练（秒/步） | 测量（秒/轮） | 总耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 4070 Ti | JAX 原版 | 1.44 | 18.47 | 49.44 | 7.5 小时 |
| RTX 4070 Ti | PyTorch 通用版 | 1.52 | 18.44 | 37.80 | 7.1 小时 |
| RTX 4070 Ti | PyTorch 优化版 | 0.79 | 8.07 | 13.95 | 3.1 小时 |
| RTX 5060 Ti | JAX 原版 | 2.44 | 30.20 | 83.28 | 12.4 小时 |
| RTX 5060 Ti | PyTorch 优化版 | 1.00 | 9.46 | 21.09 | 3.8 小时 |

## PyTorch 实现优化

| 环节 | 改动 |
| --- | --- |
| 交换采样 | 用 CUDA Graph 重放交换提议。对称性采样跳过同向自旋交换所造成的重复波函数计算。 |
| 局域能量 | 在设备上筛选有效交换，分批计算波函数。对称性路径用 CUDA Graph 重放批量前向并复用捕获池。 |
| 周期卷积 | 直接按周期坐标取值，合并部分激活、残差和归约操作。 |
| MinSR | 批量计算并缓存逐样本导数。符合条件时只计算 Gram 矩阵的一个三角，实值默认配置仍用 FP64 构造 Gram 矩阵和求解。 |
| 绘图 | 前 100 步每 10 步画一次，此后每 100 步画一次，最后一步也出图。 |

专用 CUDA 内核按设备、网络形状和运行环境启用，其余配置使用通用实现。[CUDA 开关](pytorch-migration-plan.md#cuda-开关)

`nsamples` 是采样链数。`max_parallel` 限制单次前向的输入数，对称性训练和测量的 CUDA 前向批量取它与 `projected_forward_batch` 的较小值，按展开后的对称轨道成员计数。

## JAX：设备端局域能量

三个 JAX 训练和测量示例默认启用 `Heisenberg(device_local=True)`。通用路径在 CPU 上用 QuSpin 生成交换后的构型。启用设备端路径后，筛选可交换的键、生成构型、分批计算波函数振幅和累加贡献都在 JAX 当前设备上完成。用 GPU 运行时，这省去了每次计算局域能量时在 CPU 和 GPU 之间搬运构型的开销。设备端路径沿用原版的局域能量和 MinSR 更新公式，适用于符合条件的单设备 `Variational` 自旋态。[Heisenberg 实现](../quantax/operator/common_operators.py)

## 验证命令

[JAX 检查脚本](../scripts/check_jax.py) · [PyTorch 检查脚本](../scripts/check_torch.py)

```bash
JAX_PLATFORMS=cpu environments/py311-jax062/.venv/bin/python -m scripts.check_jax
JAX_PLATFORMS=cuda environments/py311-jax062/.venv/bin/python -m scripts.check_jax
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.check_torch --cuda-fast-paths
```
