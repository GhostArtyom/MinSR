# MinSR 实现改进

## 数值一致性验证

PyTorch 优化沿用原版的 Heisenberg 算符、Metropolis 采样和 MinSR 更新公式。固定权重与构型的数值对照已通过。浮点运算顺序仍可能影响采样的接受判定，长期训练的能量和方差按统一容差验收。[完整验收结果](pytorch-migration-plan.md#完整验收结果)

## JAX 与 PyTorch 耗时对比

下表使用 10×10 方格、16 层、16 通道、3×3 卷积和 1000 个样本的完整运行。单步或单轮耗时取第 11 步或第 11 轮起的中位数，训练另排除绘图和存盘的步。总耗时为三阶段的实测耗时之和，含热化、训练绘图和存盘。JAX 基线来自启用设备端局域能量前的归档，PyTorch 版本为 `2.10.0+cu128`。[各阶段日志](reproduction-results.md#已完成的复现) · [各阶段实测与全流程估算](pipeline-runtime-estimates.md)

| 设备 | 实现 | 普通训练（秒/步） | 对称性训练（秒/步） | 测量（秒/轮） | 总耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 4070 Ti | JAX | 1.44 | 18.47 | 49.44 | 7.5 小时 |
| RTX 4070 Ti | PyTorch | 0.79 | 8.07 | 13.95 | 3.1 小时 |
| RTX 5060 Ti | JAX | 2.44 | 30.20 | 83.28 | 12.4 小时 |
| RTX 5060 Ti | PyTorch | 1.00 | 9.46 | 21.09 | 3.8 小时 |

## PyTorch 实现优化

| 环节 | 改动 |
| --- | --- |
| [交换采样](../quantax_torch/sampler/metropolis.py) | 用 CUDA Graph 重放交换提议。对称性采样在 GPU 上筛出异向自旋的提议，跳过同向交换的波函数求值。采样链超出单图容量时分组复用图，保留原随机数生成顺序。 |
| [局域能量](../quantax_torch/operator/common_operators.py) | 在设备上分块筛选有效交换，分批求波函数并累加。对称性路径按设备计时选择前向批量，复用整批的 CUDA Graph 和缓冲区，余下交换走即时计算。 |
| [卷积前向](../quantax_torch/nn/_periodic_cuda.py) | 普通训练采用 `channels_last` 存储布局。专用内核用 IEEE FP32 直接按周期坐标取值，省去显式填充，并在支持的层合并激活、残差加法或末端归约。 |
| [卷积反向](../quantax_torch/optimizer/_res_sum.py) | 对支持的 ResSum 块批量计算逐样本导数。普通训练按设备实测选择周期卷积反向或 `unfold`，对称性专用反向直接归约同一样本的轨道成员。 |
| [MinSR 导数分批与缓存](../quantax_torch/optimizer/minsr.py) | 按显存余量缓存中心化和加权后的导数或原生 FP32 导数，求解后只重算未缓存的块。 |
| [MinSR Gram 矩阵](../quantax_torch/optimizer/_cuda_gram.py) | 对符合条件的实值 FP64 矩阵，在当前设备计时比较全矩阵乘法、三角分块和 cuBLAS `SYRK`。三角路径计算一侧再补齐。 |
| [MinSR 特征值求解](../quantax_torch/optimizer/solver.py) | 对大型 FP64 矩阵用归约检查有限值，省去同尺寸布尔掩码。可用显存偏低时，按需释放空闲的 CUDA 分配器缓存。训练可指定 cuSOLVER 或 MAGMA 后端偏好。 |

专用 CUDA 内核按网络形状和 Triton 的设备及运行环境要求启用。周期卷积反向也会在 Triton 编译失败时回退。训练日志在同一行记录导数后端、缓存和每批样本数，另列 Gram 后端。[CUDA 开关](pytorch-migration-plan.md#cuda-开关)

`nsamples` 是采样链数。`max_parallel` 限制单次前向的输入数，按展开后的对称轨道成员计数。设置 `projected_forward_batch` 可在该上限内固定对称性前向批量。`jacobian_batch` 按求导的样本构型计数，中心化、Gram 矩阵和求解都使用全部样本。

PyTorch 在未指定 `jacobian_batch` 时，将 GPU 空闲显存和分配器中可复用的缓存计入预算，按进程的显存上限截断。扣除导数输出、FP64 中心化临时量和设备总显存的 10% 后，再按网络和对称轨道估算每批可处理的构型数。单进程 CUDA 更新遇到显存不足时，先释放分配器中未使用的缓存，再缩减导数缓存或批量，用原样本重试。 [PyTorch MinSR 实现](../quantax_torch/optimizer/minsr.py)

## JAX：设备端局域能量

三个 JAX 训练和测量示例默认启用 `Heisenberg(device_local=True)`。通用路径在 CPU 上用 QuSpin 生成交换后的构型。启用设备端路径后，筛选可交换的键、生成构型、分批计算波函数振幅和累加贡献都在 JAX 当前设备上完成。用 GPU 运行时，这省去了每次计算局域能量时在 CPU 和 GPU 之间搬运构型的开销。设备端路径沿用原版的局域能量和 MinSR 更新公式，适用于符合条件的单设备 `Variational` 自旋态。[Heisenberg 实现](../quantax/operator/common_operators.py)

## JAX 导数分批

单 GPU 上的 JAX 对称性训练在未指定 `jacobian_batch` 时，先按分配器余量估算导数批量，为完整 Gram 矩阵、特征分解和导数输出预留显存。导数执行遇到显存不足时可缩小批量，中心化和求解仍使用全部样本。若分配初始 Gram 矩阵时显存不足，直接报错，因为矩阵尺寸由总样本数决定。 [JAX MinSR 实现](../quantax/optimizer/minsr.py)

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
