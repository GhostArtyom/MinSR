# MinSR 的 PyTorch 移植说明

[`quantax_torch/`](../quantax_torch/) 是本仓库 `quantax` 的 PyTorch 实现。[`examples_torch/`](../examples_torch/) 的普通训练、对称性训练和固定权重测量可独立运行，也可由 `run_pipeline` 串联。原 JAX 入口继续保留。10×10 方格的端到端验收已覆盖 RTX 4070 Ti 和 RTX 5060 Ti，初始权重和验收统计见[完整验收结果](#完整验收结果)。

## 安装和运行

按 [README 的 PyTorch 安装说明](../README.md#python-311--pytorch-2100) 构建 QuSpin 并同步环境。以下命令从仓库根目录运行。

常用配置都在 [`examples_torch/run_pipeline.py`](../examples_torch/run_pipeline.py) 开头，不需要单独的配置文件。`Config` 设置共享的晶格、网络和批量；`STAGES` 设置阶段长度，也可单独覆盖 `nsamples`、`max_parallel`、`thermal_steps` 和 `sweep_steps`。保持三阶段的晶格和网络一致。编辑后直接运行：

```bash
.venv/bin/python -m examples_torch.run_pipeline
```

也可以用命令行指定设备、种子、样本数和三个阶段的长度：

```bash
.venv/bin/python -m examples_torch.run_pipeline \
  --device cuda --seed 2347 --nsamples 1000 \
  --steps 2000 --symm-steps 1000 --measure-steps 100
```

默认体系是 10×10 周期方格，$J_1=1$、$J_2=0.5$，`msr=True`，`ResSum(16, 16, 3, use_sinh=True)`。每阶段 1000 个样本，`max_parallel=32768`，交换采样使用第一和第二近邻。

| 阶段 | 默认长度 | 对称约束 | `reweight` | 热化 / 交换 | 参数步长 |
| --- | ---: | --- | ---: | ---: | --- |
| 普通训练 | 2000 步 | `TotalSz(0)` | 1.0 | 2000 / 200 | 0.001 |
| 对称性训练 | 1000 步 | `TotalSz(0) + SpinInverse() + C4v()` | 1.0 | 2000 / 200 | 0.001 |
| 测量 | 100 轮 | `TotalSz(0) + SpinInverse() + C4v()` | 2.0 | 10000 / 1000 | — |

普通训练默认随机初始化。传入 `--initial-weights` 可从指定的 `.eqx` 或 `.pt` 权重起跑，执行全部阶段。已有普通训练最终权重时，用 `FROM_CHECKPOINT` 或 `--from-checkpoint` 直接开始对称性训练。单独运行对称性训练或测量时，用 `examples_torch.train_symm --param-file` 或 `examples_torch.measure --param-file`。测量也可用 `--network-idx` 选择内置网络预设。[对称性训练入口](../examples_torch/train_symm.py) · [测量入口](../examples_torch/measure.py)

普通训练和对称性训练沿用原版 MinSR 的中心化、伪逆和固定步长。训练遇到非有限值或求解失败时停止。

结果默认写到 `runs/torch/<system>/<timestamp>/`。训练输出日志、PDF 和权重，默认最终权重为 `params_2000.pt` 和 `params_symm_1000.pt`。测量输出 `measurement.log`、`measurement.csv` 和 `summary.json`。

### CUDA 开关

可在 `Config` 设置，也可以用对应的 `--no-...` 选项单独关闭一个优化做对照。

| 配置 | 用途 |
| --- | --- |
| `projected_forward_batch=1024` | 对称性训练和测量的前向批量，按展开后的对称轨道成员计数，同时受 `max_parallel` 限制。 |
| `projected_periodic_conv` / `ordinary_periodic_conv` | 分别控制对称阶段和普通训练的专用周期卷积前向。 |
| `projected_local_energy_graph` | 对称阶段的局域能量整批前向重放。 |
| `projected_exchange_graph` | 对称阶段的压缩交换提议重放。 |

专用内核按设备和网络配置启用，其余配置使用通用路径。两条对称性 CUDA Graph 路径依赖周期卷积前向。实际启用的路径可查训练日志和测量摘要。[实现改进](implementation-improvements.md#pytorch-实现优化)

## 数值验收

在仓库根目录用 JAX 导出固定权重和构型，再用 Torch 读取同一份输入。[数值对照脚本](../scripts/check_torch.py)

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  environments/py311-jax062/.venv/bin/python -m scripts.check_torch --export-jax runs/torch-reference
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.check_torch --reference runs/torch-reference
```

4×4 方格的 JAX 和 PyTorch 固定输入对照以及 PyTorch CUDA 检查均通过。比对范围包括波函数、局域能量、Jacobian、Gram 矩阵和 MinSR 更新。两个训练入口、测量入口和三阶段流水线也通过了短测。

PyTorch 环境中的 QuSpin 接口也通过了 4×4 方格的 CPU 检查，覆盖对称基底上的稠密态转换、精确对角化、精确采样和 `TDVP_exact` 更新。

### 完整验收结果

10×10 方格的 16 通道配置依次执行 2000 步普通训练、1000 步对称性训练和 100 轮固定权重测量。对称性训练和测量各自读取上一阶段的最终权重，PyTorch 各组使用种子 2347。表中每格依次列出 Pauli 总能量和中心方差，训练取末 200 步均值。[权重和日志归档](reproduction-results.md#已完成的复现) · [耗时对比](implementation-improvements.md#jax-与-pytorch-耗时对比)

| 设备 | 实现 | 初始权重 | 普通训练 | 对称性训练 | 最终测量 |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 4070 Ti | JAX 原版 | JAX 随机初始化 | −197.61995 / 20.39306 | −198.69231 / 6.21788 | −198.78751226 / 6.20959269 |
| RTX 4070 Ti | PyTorch 优化版 | PyTorch 随机初始化 | −197.61237 / 23.40521 | −198.75008 / 6.80532 | −198.80542778 / 5.82108771 |
| RTX 5060 Ti | JAX 原版 | JAX 随机初始化 | −198.08309 / 21.61546 | −198.79243 / 7.52806 | −198.81543814 / 5.23723064 |
| RTX 5060 Ti | PyTorch 优化版 | JAX CUDA 第 0 步权重 | −197.76580 / 18.77921 | −198.81113 / 6.72867 | −198.83452640 / 4.75737444 |

两组 PyTorch 运行均通过。普通训练和对称性训练以两组 JAX 结果的区间为准，总能量分别向两端扩展 0.5 和 0.2，中心方差均向两端扩展 30%。最终测量以 RTX 5060 Ti 的 JAX 归档为准，总能量绝对差须在 0.10 以内，中心方差绝对差须在 1.5 以内。

PyTorch 原生初始化的部分运行曾在训练中出现非有限值。固定故障权重和构型后，JAX 原版和 PyTorch 通用路径也能复现异常更新。RTX 5060 Ti 上通过验收的运行使用 JAX 在 CUDA 上以种子 2347 生成的初始权重，经 `--initial-weights` 传入。[普通训练日志](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_153201/training.log)

## 功能边界

- Torch 保留 `Variational` 的批量求值和态组合、算符组合及可替换求解器。原来的 JAX 转换协议以及 NetKet/jVMC 适配留在 JAX 侧。主流程通过短测不代表所有教程、对称扇区或多设备配置均已验收。
- 历史 Flax ResNet1、未随本仓库发布的入口和数据，以及单步 Lanczos、零方差外推、能隙拟合、结构因子和关联比，不在这次移植的验收范围内。

[原 JAX 流程](../examples/run_pipeline.py) · [项目总览](project-overview.md) · [教程指南](tutorial-guide.md)
