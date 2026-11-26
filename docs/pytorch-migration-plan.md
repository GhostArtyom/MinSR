# MinSR 的 PyTorch 移植说明

[`quantax_torch/`](../quantax_torch/) 是本仓库 `quantax` 的 PyTorch 实现。[`examples_torch/`](../examples_torch/) 的普通训练、对称性训练和固定权重测量可独立运行，也可由 `run_pipeline` 串联。原 JAX 入口继续保留。10×10 方格的端到端验收已覆盖 RTX 4070 Ti 和 RTX 5060 Ti，初始权重和验收统计见[完整验收结果](#完整验收结果)。

## 安装和运行

先同步 PyTorch 环境。需要精确对角化等功能时，再单独安装本机编译的 QuSpin wheel。以下命令从仓库根目录运行。[安装说明](../README.md#python-311--pytorch-2100)

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

普通训练默认随机初始化。使用 CUDA 时，卷积权重直接在 GPU 上生成，初始化和采样各自使用以 `--seed` 设置的生成器。使用 CPU 时，权重在 CPU 上生成。传入 `--initial-weights` 可从指定的 `.eqx` 或 `.pt` 权重起跑，执行全部阶段。已有普通训练最终权重时，用 `FROM_CHECKPOINT` 或 `--from-checkpoint` 直接开始对称性训练。单独运行对称性训练或测量时，用 `examples_torch.train_symm --param-file` 或 `examples_torch.measure --param-file`。测量也可用 `--network-idx` 选择内置网络预设。[对称性训练入口](../examples_torch/train_symm.py) · [测量入口](../examples_torch/measure.py)

普通训练和对称性训练沿用原版 MinSR 的中心化、伪逆和固定步长。训练遇到非有限值或求解失败时停止。

训练入口及流水线可传 `--reweight 2.0`，让训练按 $|\psi|^2$ 采样。默认 `--reweight 1.0` 沿用公开训练示例。对称性短跑可用 `--reweight 2.0 --diagnostics --param-file` 从指定权重开始，采样链会重新热化。两个采样指数使用同一 MinSR 求解流程，固定权重测量使用 2.0。[训练入口](../examples_torch/train_symm.py) · [采样器](../quantax_torch/sampler/metropolis.py)

JAX 和 PyTorch MinSR 均用 `jacobian_batch` 设置每批求导的样本数，命令行参数为 `--jacobian-batch`。PyTorch CUDA 示例优先读取完全匹配的记录。没有匹配记录时，从相同 GPU 型号、执行路径和训练阶段的相近配置取起点，较大的网络会下调候选值。没有相近记录时，按当前显存估算。估算偏低时，训练示例从最多 32 个样本起试，受总样本数和单批前向可容纳的样本数限制。首步用热化后的实际样本验证批量，其更新直接用于训练。首步日志用 `samples/batch` 记录实际导数批量。前向用 `inputs`，采样图分组用 `chains/group`，对称性局域能量用 `orbit members/batch`。

普通训练、对称性训练和测量分别保存记录。记录按 GPU、PyTorch/CUDA 版本、网络、体系、样本数和后端配置匹配。训练记录包含验证通过的雅可比批量，以及已选出的 Gram 后端和局域能量前向批量。显存不足时用原样本回退，既有批量记录继续用于后续运行。普通训练默认比较每批 512、1024 和 2048 个连接构型，受 `max_parallel` 限制。对称性训练和测量在未显式指定 `projected_forward_batch` 且支持 CUDA Graph 时，也用热化后的样本选前向批量。文件默认保存在 `~/.cache/minsr/torch-cuda-tuning/`，设置 `XDG_CACHE_HOME` 后以该目录为根。日志记录使用的 JSON 路径，删除该文件可重新验证批量。 [CUDA 调优实现](../examples_torch/_cuda_tuning.py)

可以先用独立入口为指定网络和样本数生成记录。任选 `train`、`train_symm` 或 `measure`，每次按该阶段的配置热化并取一批样本，权重固定。以下用仓库附带的权重为测量阶段提前调优。省略 `--param-file` 时在 CUDA 上随机初始化，也可以传入自己的检查点。

```bash
.venv/bin/python -m examples_torch.tune_cuda measure --depth 30 --channels 64 --nsamples 1000 \
  --param-file params/square10x10_3064.eqx
```

需要比较雅可比和局域能量批量的组合时，用独立的 [CUDA 矩阵基准](../scripts/benchmark_torch_training.py)。默认每组使用固定样本预热 2 次、计时 5 次，再将每阶段最快的两组各用 2 步/轮预热和 10 步/轮计时做完整复测。结果写入 `runs/torch/<system>/<timestamp>/` 下的 `benchmark.log` 和 `summary.json`。加上 `--save-tuning` 时，复测通过的胜者写入自动调优缓存。

```bash
.venv/bin/python -m scripts.benchmark_torch_training --matrix --depth 30 --channels 64 --nsamples 1000 --save-tuning
```

对比采样指数时，单独运行训练基准两次，分别传入 `--reweight 1.0` 和 `--reweight 2.0`。用同一个 `--param-file`、随机种子、样本数和批量设置。加上 `--symmetric` 可测投影后的训练。终端逐步输出采样、局域能量、MinSR 和完整更新的耗时，以及能量、中心方差、参数更新范数和重加权有效样本数。耗时中位数排除预热和热化，重加权有效样本数只反映权重集中程度。[训练基准](../scripts/benchmark_torch_training.py)

已有候选时，可以按 [批量实测与选择](pytorch-cuda-benchmark.md) 交替短测，记录逐步的分配峰值、预留峰值和驱动空闲量。WSL 下也可同时采样 Windows 的专用及共享 GPU 内存。

单进程 CUDA MinSR 更新遇到显存分配错误时，用原样本缩减导数缓存或批量后重试。中心化、Gram 矩阵和求解始终使用全部样本。 [MinSR 实现](../quantax_torch/optimizer/minsr.py)

结果默认写到 `runs/torch/<system>/<timestamp>/`。训练输出日志、PDF 和权重，默认最终权重为 `params_2000.pt` 和 `params_symm_1000.pt`。测量输出 `measurement.log`、`measurement.csv` 和 `summary.json`。各阶段在热化开始、每完成 100 次交换提议以及热化结束时写入进度，后两列分别为距上次报告的区间耗时和阶段累计耗时，单位均为秒。

临时测试用 `--output-dir runs/<测试名>` 指定独立的输出根目录。

### CUDA 开关

可在 `Config` 设置。命令行参数使用小写连字符，布尔开关可加 `--no-` 前缀关闭。

| 配置 | 用途 |
| --- | --- |
| `projected_forward_batch` | 按展开后的对称轨道成员固定对称性前向批量，同时受 `max_parallel` 限制。默认采样用 1024，局域能量按设备实测选择 1024、2048 或 4096。 |
| `local_energy_forward_batch` | 单独固定局域能量的前向批量，按轨道成员计数，须容纳整数个轨道且至多为 `max_parallel`。同时设置 `projected_forward_batch` 时，采样使用自己的批量。 |
| `projected_periodic_conv` / `ordinary_periodic_conv` | 分别控制对称阶段和普通训练的专用周期卷积前向。 |
| `projected_local_energy_graph` | 对称阶段的局域能量整批前向重放。 |
| `projected_exchange_graph` | 对称阶段的压缩交换提议重放。 |
| `eigh_backend` | 训练进程的 CUDA 线性代数后端偏好。`default` 采用 PyTorch 默认分派，`cusolver` 和 `magma` 分别优先使用 cuSOLVER 和 MAGMA。省略时沿用进程原设定，日志显示当前偏好。大型 FP64 求解在可用显存偏低时可释放空闲缓存，显式选择 MAGMA 时跳过这一步。 |

周期卷积、对称性局域能量重放和压缩交换提议重放在支持 Triton 的 NVIDIA CUDA 设备（计算能力 8.0 起）上按网络形状和精度启用。普通训练首次按设备计时，在周期卷积反向和 `unfold` 之间选路。符合条件的实值 FP64 Gram 在当前设备上计时选择全矩阵、三角分块或 `SYRK`。局域能量复用整批的 CUDA Graph，余数走普通前向。采样链超出单图容量时分组重放。实际路径可查训练日志和测量摘要。 [实现改进](implementation-improvements.md#pytorch-实现优化)

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
| RTX 4070 Ti | JAX | JAX 随机初始化 | −197.61995 / 20.39306 | −198.69231 / 6.21788 | −198.78751226 / 6.20959269 |
| RTX 4070 Ti | PyTorch | PyTorch 随机初始化 | −197.61237 / 23.40521 | −198.75008 / 6.80532 | −198.80542778 / 5.82108771 |
| RTX 5060 Ti | JAX | JAX 随机初始化 | −198.08309 / 21.61546 | −198.79243 / 7.52806 | −198.81543814 / 5.23723064 |
| RTX 5060 Ti | PyTorch | PyTorch CUDA 随机初始化 | −197.79088 / 20.12901 | −198.72981 / 5.55081 | −198.79906050 / 4.94921107 |

两组 PyTorch 运行均通过。普通训练和对称性训练以两组 JAX 结果的区间为准，总能量分别向两端扩展 0.5 和 0.2，中心方差均向两端扩展 30%。最终测量以 RTX 5060 Ti 的 JAX 归档为准，总能量绝对差须在 0.10 以内，中心方差绝对差须在 1.5 以内。

PyTorch 早期在 CPU 上随机生成初始权重的部分运行曾出现非有限值。固定故障权重和构型后，JAX 和 PyTorch 通用路径也能复现异常更新。表中的 RTX 5060 Ti PyTorch 运行由 PyTorch 在 CUDA 上以种子 2347 生成初始权重。[训练不稳定性排查](minsr-training-instability.md)

## 功能边界

- Torch 保留 `Variational` 的批量求值和态组合、算符组合及可替换求解器。原来的 JAX 转换协议以及 NetKet/jVMC 适配留在 JAX 侧。主流程通过短测不代表所有教程、对称扇区或多设备配置均已验收。
- 历史 Flax ResNet1、未随本仓库发布的入口和数据，以及单步 Lanczos、零方差外推、能隙拟合、结构因子和关联比，不在这次移植的验收范围内。

[原 JAX 流程](../examples/run_pipeline.py) · [项目总览](project-overview.md) · [教程指南](tutorial-guide.md)
