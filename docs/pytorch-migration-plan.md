# MinSR 的 PyTorch 移植说明

[`quantax_torch/`](../quantax_torch/) 是本仓库 `quantax` 的 PyTorch 实现。[`examples_torch/`](../examples_torch/) 的普通训练、对称性训练和固定权重测量可以独立运行，也可以由 `run_pipeline` 串起来。原来的 JAX 入口继续保留。10×10 方格的默认配置已完成[端到端验收](#完整验收结果)。[JAX 与 PyTorch 耗时对比](implementation-improvements.md#jax-与-pytorch-耗时对比)

## 安装和运行

从仓库根目录执行。锁定环境为 Python 3.11、PyTorch 2.10.0/CUDA 12.8；用 `--device cpu` 可在 CPU 上运行。

```bash
uv sync --project environments/py311-torch --locked
```

常用配置都在 [`examples_torch/run_pipeline.py`](../examples_torch/run_pipeline.py) 开头，不需要单独的配置文件。`Config` 设置共享的晶格、网络和批量；`STAGES` 设置阶段长度，也可单独覆盖 `nsamples`、`max_parallel`、`thermal_steps` 和 `sweep_steps`。保持三阶段的晶格和网络一致。编辑后直接运行：

```bash
environments/py311-torch/.venv/bin/python -m examples_torch.run_pipeline
```

也可以用命令行指定设备、种子、样本数和三个阶段的长度：

```bash
environments/py311-torch/.venv/bin/python -m examples_torch.run_pipeline \
  --device cuda --seed 2347 --nsamples 1000 \
  --steps 2000 --symm-steps 1000 --measure-steps 100
```

默认体系是 10×10 周期方格，$J_1=1$、$J_2=0.5$，`msr=True`，`ResSum(16, 16, 3, use_sinh=True)`。每阶段 1000 个样本，`max_parallel=32768`，交换采样使用第一和第二近邻。

| 阶段 | 默认长度 | 对称扇区 | `reweight` | 热化 / 交换 | 参数步长 |
| --- | ---: | --- | ---: | ---: | --- |
| 普通训练 | 2000 步 | `TotalSz(0)` | 1.0 | 2000 / 200 | 0.001 |
| 对称性训练 | 1000 步 | `TotalSz(0) + SpinInverse() + C4v()` | 1.0 | 2000 / 200 | 0.001 |
| 测量 | 100 轮 | 同对称性训练 | 2.0 | 10000 / 1000 | 不更新 |

普通训练默认从随机初始化开始。流水线为每个阶段启动独立进程，下一阶段读取上一阶段实际生成的最终权重并重新热化。阶段失败时停止。要从普通训练权重开始做对称性训练和测量，设置 `FROM_CHECKPOINT` 或传 `--from-checkpoint`，跳过普通训练。只跑对称性训练用 `examples_torch.train_symm --param-file`；只测量用 `examples_torch.measure --param-file`。测量也可用 `--network-idx` 选择三个内置预设 `1616`、`1632`、`3064`。

本仓库的 `ResSum` 可读取 Equinox `.eqx`。导入会检查参数和 buffer 的形状、dtype 及文件结尾，并导入 `theta0`。按 checkpoint 设置网络结构；Torch 新写出的 checkpoint 为带元数据和版本号的 `.pt`，默认最终文件为 `params_2000.pt` 和 `params_symm_1000.pt`。

结果默认写到 `runs/torch/<system>/<timestamp>/`。训练保留 `training.log` 或 `training_symm.log`、PDF 和 checkpoint，每 100 步及最后一步存权重。测量写 `measurement.log` 和逐轮刷新的 `measurement.csv`，在最后一轮完成后才写 `summary.json`。日志记录配置、后端和设备、种子及输入权重路径。

### CUDA 开关

可在 `Config` 设置，也可以用对应的 `--no-...` 选项单独关闭一个优化做对照。

| 配置 | 用途 |
| --- | --- |
| `projected_forward_batch=1024` | 对称性训练和测量的前向批量，按展开后的对称轨道成员计数，同时受 `max_parallel` 限制。 |
| `projected_periodic_conv` / `ordinary_periodic_conv` | 分别控制对称阶段和普通训练的专用周期卷积前向。 |
| `projected_local_energy_graph` | 对称阶段的局域能量整批前向重放。 |
| `projected_exchange_graph` | 对称阶段的压缩交换提议重放。 |

不满足专用内核的设备或形状要求时走通用路径。关闭专用周期卷积也会使依赖它的两条对称 CUDA Graph 路径回退。要只比较某一张图，用对应的开关。测量的 `summary.json` 记录实际前向批量、是否使用专用周期前向以及两张 CUDA Graph 是否执行。训练日志也记录权重梯度、导数缓存和 Gram 后端。[CUDA 设备适配与默认启用范围](implementation-improvements.md#cuda-设备适配与性能)

## 数值验收

在仓库根目录用 JAX 导出固定权重和构型，再用 Torch 读取同一份输入：

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.check_torch --export-jax runs/torch-reference
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  environments/py311-torch/.venv/bin/python -m scripts.check_torch --reference runs/torch-reference
```

4×4 方格的 CPU 对照已通过。一次复查中，逐样本局域能量和 MinSR 更新的最大绝对差分别为 $2.74\times10^{-7}$ 和 $3.88\times10^{-6}$。三个独立入口通过了无 QuSpin 的 CPU 短测；另有 CPU 检查验证两进程分片与单进程的更新一致，不能代替双 GPU 验收。CUDA 数值检查和三阶段各 2 步的流水线短测也已通过。

完整验收用下面两组 16 通道的 JAX 归档作为参考。训练统计取各阶段末 200 步的平均总能量和中心方差，每步使用更新前的样本。测量均为 100 轮，固定对应的对称性训练权重；B 采用同一权重的 100 轮重测结果。

| JAX 归档 | 普通训练：总能量 / 方差 | 对称性训练：总能量 / 方差 | 测量：总能量 / 每格点能量 / 方差 |
| --- | ---: | ---: | ---: |
| A，RTX 5060 Ti | −198.08309 / 21.61546 | −198.79243 / 7.52806 | [−198.81543814 / −1.98815438 / 5.23723064](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/summary.json) |
| B，RTX 4070 Ti | −197.61995 / 20.39306 | −198.69231 / 6.21788 | [−198.78751226 / −1.98787512 / 6.20959269](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261120_022539/summary.json) |

归档目录、能量换算和另一次 32 通道的完整结果见[复现记录](reproduction-results.md#已完成的复现)。

训练的中心方差取自日志，测量的中心方差按 CSV 的原始二阶矩均值减去能量均值的平方重算。下文的能量和方差均采用 Pauli 哈密顿量口径。

### 完整验收结果

2026 年 11 月 21 日，PyTorch 通用版在 RTX 4070 Ti 上读取归档 A 的 `params_symm_1000.eqx` 独立测量 100 轮。总能量为 -198.81902330，每格点能量为 -1.98819023，中心方差为 5.66303716。以连续 10 轮为块做 jackknife，相对 A 的总能量和方差绝对差分别为 0.32 和 1.05 个合并标准误。[同权重测量摘要](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_022743/summary.json) · [逐轮数据](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_022743/measurement.csv)

PyTorch 通用版和 `sm_89` 优化版都以种子 2347 从随机初始化完成 2000 步普通训练、1000 步对称性训练和 100 轮最终测量。优化版只跑流水线，最终测量读取本次生成的对称性权重。验收检查了阶段间权重交接、最终权重的有限值和 PDF 内嵌的 Arial 字体，并在优化版日志中确认周期前向、普通训练的 FP64 下三角 Gram 及两个对称阶段的 CUDA Graph 实际启用。

| 阶段 | 通用版总能量 / 中心方差 | 优化版总能量 / 中心方差 | 优化版相对 A | 优化版相对 B |
| --- | ---: | ---: | ---: | ---: |
| 普通训练末 200 步 | [-197.78286 / 21.76796](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_033121/training.log) | [-197.61237 / 23.40521](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_123654/training.log) | +0.47072 / +1.78974 | +0.00759 / +3.01215 |
| 对称性训练末 200 步 | [-198.77327 / 6.03977](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_042306/training_symm.log) | [-198.75008 / 6.80532](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_130358/training_symm.log) | +0.04235 / -0.72274 | -0.05777 / +0.58744 |
| 最终测量 100 轮 | [-198.82015 / 5.16789](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_093024/summary.json) | [-198.80543 / 5.82109](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_152043/summary.json) | +0.01001 / +0.58386 | -0.01792 / -0.38850 |

两轮均满足工程容差。普通训练和对称性训练的总能量按 A/B 区间分别向两端扩展 0.5 和 0.2，中心方差区间向两端扩展 30%。最终测量以 A 为基准，总能量容差为 0.10（每格点 0.001），中心方差容差为 1.5。优化版最终每格点能量为 -1.98805428。优化版对称性训练日志缺第 620–628 步记录，末 200 步、最终权重和测量均齐全。两轮各使用一个相同的 Torch 随机种子。[各阶段归档](reproduction-results.md#已完成的复现) · [耗时对比](implementation-improvements.md#jax-与-pytorch-耗时对比)

## 功能边界

- 主流程不依赖 QuSpin。稠密对称基底、精确对角化、精确采样和 `TDVP_exact` 仍需可选的 QuSpin 接口；需要时在 Torch 环境安装 `--extra ed` 的辅助依赖和兼容的 `cp311` QuSpin wheel。
- Torch 保留 `Variational` 的批量求值和态组合、算符组合及可替换求解器。原来的 JAX 转换协议以及 NetKet/jVMC 适配留在 JAX 侧。主流程通过短测不代表所有教程、对称扇区或多设备配置均已验收。
- 历史 Flax ResNet1、未随本仓库发布的入口和数据，以及单步 Lanczos、零方差外推、能隙拟合、结构因子和关联比，不在这次移植的验收范围内。

[原 JAX 流程](../examples/run_pipeline.py) · [项目总览](project-overview.md) · [教程指南](tutorial-guide.md)
