# MinSR 项目概览

## 项目定位与适用范围

MinSR 是论文 *Empowering deep neural quantum states through efficient optimization* 的配套仓库。它用神经量子态（Neural Quantum State，NQS）近似量子多体波函数，并提供一套从晶格建模、对称性约束、蒙特卡洛采样到参数优化和能量测量的实现。项目背景和论文入口见根目录的 [README](../README.md)。

这里的神经网络接收一个自旋构型，输出该构型的波函数振幅。训练目标是降低给定哈密顿量的变分能量。代码围绕自旋晶格模型展开，教程覆盖横场 Ising 模型、方格 Heisenberg 模型，以及方格和三角格上的受挫 $J_1$-$J_2$ 模型。

仓库名称是 MinSR，实际导入的 Python 包名是 `quantax`。MinSR 是其中的优化算法；Quantax 还负责晶格、算符、量子态、网络、对称性和采样。根目录说明将其定位为论文所用的历史版本源码，使用和修改时应以本仓库接口为准。

## 变分能量与训练流程

设自旋构型为 $s=(s_1,\ldots,s_N)$，其中 $s_i\in\{-1,1\}$。网络参数为 $\theta$，输出为 $\psi_\theta(s)$。目标是最小化归一化后的能量期望：

$$
E(\theta)=\frac{\sum_{s,s'}\psi_\theta^*(s)H_{s,s'}\psi_\theta(s')}{\sum_s|\psi_\theta(s)|^2}.
$$

完整波函数的分量数随自旋数指数增长。小系统可以枚举基底或精确对角化；训练较大系统时，程序从当前波函数采样，利用局域能量估计能量和优化所需的量：

$$
E_{\mathrm{loc}}(s)=\frac{\sum_{s'}H_{s,s'}\psi_\theta(s')}{\psi_\theta(s)}.
$$

对应实现位于 [operator.py](../quantax/operator/operator.py) 的 `psiOloc`、`Oloc` 和 `expectation`。程序只计算哈密顿量连接到的构型上的振幅，不必在每次迭代中保存完整波函数。

一次训练按下面的顺序完成：

1. 创建晶格，确定格点数、坐标、边界条件和邻接关系。
2. 创建哈密顿量，指定相互作用和耦合常数。
3. 创建神经网络，用 `Variational` 包装成可求值、求导和更新的变分态，并指定对称性。
4. 创建采样器，从当前变分态生成自旋构型。
5. 优化器计算局域能量、波函数对参数的导数及更新方向。
6. 调用 `state.update(...)` 更新参数，记录能量、能量方差，按需保存权重。

教程和示例反复使用同一段训练结构：

```python
samples = sampler.sweep()
step = optimizer.get_step(samples)
state.update(step * 1e-3)
```

`get_step` 计算方向，`update` 执行参数更新。后者内部已经处理了更新方向的负号和整体波函数尺度，照示例传入正的步长系数即可。`optimizer.energy` 和 `optimizer.VarE` 来自这次 `get_step` 使用的样本，对应本轮更新前的状态。

## MinSR 的样本空间求解

随机重构（Stochastic Reconfiguration，SR）根据波函数在参数空间中的几何结构选择更新方向。本仓库的 `TDVP` 默认执行虚时间优化，用来降低能量；`TDVP_exact` 用完整基底上的波函数计算对应更新，适合小系统检查。

用 $N_s$ 表示样本数，$N_p$ 表示可训练参数数。对于默认的非重加权、实参数和实输出情形，可把每个样本上的对数导数及局域能量中心化，写成矩阵和向量：

$$
O_{sk}=\frac{1}{\psi_\theta(s)}\frac{\partial\psi_\theta(s)}{\partial\theta_k},\qquad \bar O_{sk}=\frac{O_{sk}-\langle O_k\rangle}{\sqrt{N_s}},\qquad \bar\epsilon_s=\frac{E_{\mathrm{loc}}(s)-E}{\sqrt{N_s}}.
$$

参数空间中的 SR 方程涉及 $S=\bar O^\dagger\bar O$，其大小为 $N_p\times N_p$。当参数远多于样本时，MinSR 转而构造样本空间矩阵：

$$
T=\bar O\bar O^\dagger,\qquad \delta\theta=\bar O^\dagger T^+\bar\epsilon.
$$

这里的 $T^+$ 表示伪逆形式的求解。实值训练示例默认用 FP32 保存网络参数，以 FP64 构造样本空间矩阵并做特征值分解。默认伪逆采用论文式 (23) 的软截断，FP64 下相对阈值为 $10^{-12}$，绝对阈值为 0。调用 `state.update(eta * step)` 后，网络参数沿 $-\eta\delta\theta$ 更新，同时调整波函数的整体尺度。[伪逆实现](../quantax/optimizer/solver.py) · [论文 Methods](Chen-2024-MinSR/paper.pdf#page=8)

这样，主要的矩阵求解从参数数决定的维度转到样本数决定的维度。当 $N_p\gg N_s$ 时，这一转换适合深网络。它仍需构造样本空间矩阵，样本数增大时也会增加内存和求解开销，不能仅靠增加样本数消除训练成本。

两种优化接口在本仓库中还有实现上的区别：

| 接口 | 导数与矩阵的处理 | 使用位置 |
| --- | --- | --- |
| `TDVP` | 先调用 `state.jacobian` 构造完整导数矩阵；默认求解器按矩阵形状选择样本空间或参数空间算法 | 5 个教程 |
| `MinSR` | 对支持的顺序网络按参数块反向计算导数并累加 $T$，求解后计算导数共轭转置与向量的乘积。按样本分批时，先汇合各批的导数块，再用全部样本计算中心化与 Gram 矩阵 | [`examples/train.py`](../examples/train.py)、[`examples/train_symm.py`](../examples/train_symm.py) |
| `TDVP_exact` | 在指定对称性基底上计算完整波函数及导数，不依赖蒙特卡洛样本 | 教程 1 |

因此，教程中的 `TDVP` 在参数多于样本时也可能使用最小范数求解。`MinSR` 类进一步改变了导数的计算和保存方式。相关源码见 [tdvp.py](../quantax/optimizer/tdvp.py)、[minsr.py](../quantax/optimizer/minsr.py) 和 [solver.py](../quantax/optimizer/solver.py)。

上述矩阵尺寸说明针对实数情形。实参数、复输出时，实现会拼接实部和虚部，样本空间维度变为 $2N_s$。`MinSR` 明确拒绝非全纯复参数态（`vs_type == 1`），对实参数、复输出的网络也要求复数中间量只出现在末尾的无参数层中。

## 网络、对称性与采样

### 波函数的网络表示

教程 1 使用全连接受限玻尔兹曼机 `RBM_Dense`，其余教程使用残差卷积网络 `ResSum`。`ResSum(depth, channels, kernel_size)` 的前三个参数分别指定卷积层数、通道数和卷积核大小，`depth` 必须是偶数。卷积采用周期填充，末端通过空间求和或带平移特征标的求和生成波函数振幅。

对 10×10 方格的实值 `ResSum`，末层逐项激活后取平均。`use_sinh=False` 使用 $\exp(x+\theta_0)$，在当前对称性配置下输出正振幅。`use_sinh=True` 改用 $\exp(\theta_0)(\sinh(x)+1)$，允许不同构型的振幅取正或取负。这里 $\theta_0$ 控制整体尺度，加 1 让 $x=0$ 时的激活保持为正。[网络实现](../quantax/nn/sum_nets.py) · [激活函数](../quantax/nn/activation.py)

`msr=True` 在哈密顿量中实现 Marshall 基底变换，翻转第一近邻的横向交换项，保留第二近邻项的符号。对 $J_2/J_1=0.5$ 的受挫方格，Marshall 规则给出近似的符号结构。启用 `use_sinh=True` 后，网络可以在这个基底里学习哪些构型需要翻号。[Heisenberg 实现](../quantax/operator/common_operators.py) · [论文的网络结构与符号说明](Chen-2024-MinSR/paper.pdf#page=8)

三角格教程使用复振幅。网络把实通道配成复数后使用指数激活，网络参数仍为实数。[网络实现](../quantax/nn/sum_nets.py)

### 对称性约束

`TotalSz()` 指定总磁化为零的子空间，`SpinInverse()` 指定自旋反演偶扇区，`SpinInverse(-1)` 指定奇扇区。`Trans1D`、`Trans2D` 指定平移动量扇区，`C4v` 和 `D6` 分别用于方格和三角格点群。`+` 用来组合这些约束。

空间对称性和自旋反演通过对变换后的构型求值并加权平均来投影波函数。固定总磁化则还依赖基底和采样器：`TotalSz` 决定精确基底中的上自旋数，交换采样在该磁化扇区内移动。默认零动量的 `ResSum` 已通过周期卷积和末端求和编码平移对称性，因此教程 2、3 在变分态外部只增加自旋反演和点群投影。实现见 [symmetry.py](../quantax/symmetry/symmetry.py) 和 [nqs_layers.py](../quantax/nn/nqs_layers.py)。

### 构型采样

Ising 模型含单自旋翻转项，教程使用 `LocalFlip`。Heisenberg 类模型的教程固定总磁化，使用交换两个格点自旋的 `NeighborExchange`。参数 `n_neighbor=[1, 2]` 允许在第一、第二近邻键上提出交换，不会改变哈密顿量本身。

默认 `reweight=2.0` 按 $|\psi|^2$ 采样。大规模训练示例改用 `reweight=1.0`，按 $|\psi|$ 采样，并通过 `Samples.reweight_factor` 把统计量重加权回目标分布。训练脚本的注释标明，该设置仅用于不超过 100 个格点的系统。实现见 [metropolis.py](../quantax/sampler/metropolis.py) 和 [status.py](../quantax/sampler/status.py)。

## 仓库结构

| 路径 | 职责或材料 |
| --- | --- |
| [quantax/global_defs.py](../quantax/global_defs.py) | 全局随机数、计算类型、参数类型和当前晶格 |
| [quantax/sites/](../quantax/sites/) | 晶格几何、周期边界和近邻关系 |
| [quantax/operator/](../quantax/operator/) | 自旋算符、常用哈密顿量、局域能量和精确对角化 |
| [quantax/symmetry/](../quantax/symmetry/) | 对称性生成、组合、波函数投影和 QuSpin 基底 |
| [quantax/nn/](../quantax/nn/) | RBM、卷积网络、残差网络和符号或相位网络 |
| [quantax/state/](../quantax/state/) | 稠密态、变分态、波函数求值、自动微分和权重读写 |
| [quantax/sampler/](../quantax/sampler/) | Metropolis 采样、翻转或交换提议、样本与重加权因子 |
| [quantax/optimizer/](../quantax/optimizer/) | TDVP、精确更新、MinSR、实时间演化和线性求解器 |
| [quantax/utils/](../quantax/utils/) | 自旋编码、数组分片、参数树操作和 `DataTracer` |
| [tutorials/](../tutorials/) | 5 个讲解模型和接口的 Notebook |
| [examples/](../examples/) | 较大系统的普通训练、对称性训练和测量脚本 |
| [params/](../params/) | 10×10 方格、$J_2/J_1=0.5$ 的 3 组已训练权重 |
| [data/](../data/) | [`Fig2a.csv`](../data/Fig2a.csv)、[`Fig2b.csv`](../data/Fig2b.csv)、[`Fig2c.csv`](../data/Fig2c.csv)、[`Fig3.csv`](../data/Fig3.csv) 图表数据 |
| [images/](../images/) | 论文相关示意图和结果图 |

JAX 负责自动微分、编译和设备计算，Equinox 负责网络模块和参数序列化，QuSpin 负责自旋基底、算符作用和精确对角化。仓库支持同一节点上的多设备计算；根目录说明仍将多节点兼容列为待实现内容。

## 训练与测量示例

只需修改 [run_pipeline.py](../examples/run_pipeline.py) 开头的配置区：`COMMON` 设置共用的晶格、网络、`NSAMPLES` 等参数，其中 `STEP_SIZE` 只用于两个训练阶段；`STAGES` 设置各阶段的 `STEPS` 等参数，也可在某个阶段单独覆盖 `NSAMPLES`。从仓库根目录直接运行，不用传参：

```bash
environments/py311-jax062/.venv/bin/python -m examples.run_pipeline
```

流水线依次执行普通训练、对称性训练和测量，并自动传递各阶段的最终权重。已有第一阶段权重时，可设置配置区的 `FROM_CHECKPOINT` 跳过普通训练；网络设置必须与权重匹配。

各阶段在热化开始、每完成 100 次交换提议以及热化结束时写入进度。进度后的两列分别为距上次报告的区间耗时和阶段累计耗时，单位均为秒。

三个阶段脚本单独运行时，都可用命令行临时覆盖各自的配置参数，未指定的参数沿用 [`run_pipeline.py`](../examples/run_pipeline.py) 中的默认值。例如用对应的 JAX 解释器运行 `python -m examples.train --STEPS 1000 --NSAMPLES 2000 --step-size 0.001`。参数名支持大写下划线或小写连字符两种写法，如 `--MAX_PARALLEL` 和 `--max-parallel`。`--use-sinh` 接受 `true` 或 `false`。各脚本的参数列表见 `--help`。

普通训练和对称性训练可用 `--jacobian-batch 1000` 将导数批量的初始上限设为 1000 个样本。对应的流水线配置为 `JACOBIAN_BATCH=1000`，直接调用 MinSR 时使用 `jacobian_batch=1000`。默认值为 `None`，单 GPU 对称性训练按显存余量估算初始批量，普通训练先用整批样本。遇到可重试的显存分配失败时，MinSR 用原样本缩批重试。每次调整都会记入日志，后续训练步沿用成功的批量。初始 Gram 矩阵的尺寸由总样本数决定，分配失败时直接报错。[流水线配置](../examples/run_pipeline.py) · [MinSR 实现](../quantax/optimizer/minsr.py)

减小导数批量会增加分批次数，可能增加内核调度开销并降低 GPU 利用率。JAX 切换批大小时通常要重新编译，浮点运算也可能产生小幅差异。中心化、Gram 矩阵和求解仍使用全部样本，因此样本数和估计量保持不变。

选批时可用 $B\approx(M_\mathrm{budget}-M_\mathrm{margin}-M_\mathrm{fixed})/m_\mathrm{sample}$ 粗估。$M_\mathrm{fixed}$ 包括完整 Jacobian、Gram 矩阵和求解工作区，$m_\mathrm{sample}$ 可由两个批大小的导数编译显存估计之差除以批大小之差。当前 16 层、16 通道的 10000 样本配置中，FP64 Jacobian 和一份 Gram 矩阵合计约 3.35 GiB。

[train.py](../examples/train.py) 在 10×10 周期方格上用 MinSR 训练 $J_1=1$、$J_2=0.5$ 的模型。每次运行在 `runs/<体系和格点名>/<运行时间>/` 保存 [`training.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_030050/training.log)、[`energy.pdf`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_030050/energy.pdf)、[`VarE.pdf`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_030050/VarE.pdf) 和 [`params_2000.eqx`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_030050/params_2000.eqx)（2000 步示例） 等文件。体系信息已包含在目录名中，文件名不再重复。日志记录运行配置、硬件、逐步能量、方差和耗时。

[train_symm.py](../examples/train_symm.py) 加入 `TotalSz() + SpinInverse() + C4v()` 约束后继续训练。脚本在独立的 `runs/<体系和格点名>/<运行时间>/` 中保存 [`training_symm.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_051106/training_symm.log)、[`energy_symm.pdf`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_051106/energy_symm.pdf)、[`VarE_symm.pdf`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_051106/VarE_symm.pdf) 和 [`params_symm_1000.eqx`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_051106/params_symm_1000.eqx)（1000 步示例） 等文件。日志开头记录运行配置和权重来源，随后以与 [`train.py`](../examples/train.py) 相同的格式记录能量、方差、完成步数及耗时，异常时记录堆栈。

单独运行对称性训练时，用对应的 JAX 解释器运行 `python -m examples.train_symm --param-file <权重文件>`，指定第一阶段的 `.eqx` 文件，并确保网络设置与权重匹配。相对路径从当前工作目录解析。

两个训练脚本每 10 步刷新能量和方差的全程图和滚动图，每 100 步保存权重；最后一步也保存曲线和权重。 [滚动能量图示例](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_030050/energy_rolling.pdf)

三个阶段都为 Heisenberg 算符启用 `device_local=True`。在单设备上，局域能量所需的自旋交换、波函数求值和求和都在同一设备完成，按 `MAX_PARALLEL` 限制前向批量。通用 QuSpin 路径仍可用，未启用该选项或不满足单设备等条件时沿用原路径。数据流和对照测试见[实现改进记录](implementation-improvements.md)。

[measure.py](../examples/measure.py) 反复采样测量能量，不更新参数。单独运行时用 `--network-idx` 选择下表中的网络，默认选 1。测量自定义网络时用 `--param-file` 指定权重，并用 `--depth`、`--channels` 指定与权重匹配的结构，不同时传 `--network-idx`：

| `network_idx` | 网络配置 | 参数文件 |
| --- | --- | --- |
| 1 | 16 层、16 通道、3×3 卷积核 | [`square10x10_1616.eqx`](../params/square10x10_1616.eqx) |
| 2 | 16 层、32 通道、3×3 卷积核 | [`square10x10_1632.eqx`](../params/square10x10_1632.eqx) |
| 3 | 30 层、64 通道、3×3 卷积核 | [`square10x10_3064.eqx`](../params/square10x10_3064.eqx) |

测量脚本的注释将这些网络对应到论文图 2b 中的 ResNet2（MinSR）结果。`.eqx` 文件保存网络权重，加载时仍须创建匹配的网络结构。单独运行时若未传 `--param-file`，则读取表中自带权重；若已修改共用的晶格尺寸、卷积核或输出层设置，使用自带权重前要恢复与预设匹配的设置。`THERMAL_STEPS` 和 `SWEEP_STEPS` 分别控制初始化热化和每轮采样的更新次数，默认按实际使用的晶格尺寸取 $100L^2$ 和 $10L^2$，也可分别覆盖。每次运行创建独立的 `runs/<体系和格点名>/<运行时间>/`，[`measurement.log`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/measurement.log) 开头记录运行配置和权重来源，随后记录累计平均能量、累计方差、完成步数与耗时；[`measurement.csv`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/measurement.csv) 逐轮保存能量、局域能量二阶矩、累计平均能量和累计方差，并立即刷新；[`summary.json`](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/summary.json) 保存最终测量结果和权重来源。中断时保留已写入的 CSV，不生成最终汇总。

## 数值约定与运行限制

### 能量与方差换算

论文式 (4) 用自旋算符 $\mathbf S$，仓库的 `Heisenberg` 用 Pauli 矩阵 $\boldsymbol\sigma$。对自旋 1/2，$\mathbf S_i=\boldsymbol\sigma_i/2$，因此每条键上的 $\mathbf S_i\cdot\mathbf S_j$ 等于 $\boldsymbol\sigma_i\cdot\boldsymbol\sigma_j/4$。相同耦合常数下，Pauli 总能量是自旋算符总能量的 4 倍，方差是 16 倍。记 $N$ 为格点数，论文每格点值与日志总量的换算为

$$
\frac{E_{\mathrm{spin}}}{N}=\frac{E_{\mathrm{Pauli}}}{4N},
\qquad
\frac{\sigma^2_{\mathrm{spin}}}{N}=\frac{\sigma^2_{\mathrm{Pauli}}}{16N}.
$$

核对训练日志的 `E` 或测量摘要的 `energy` 时用 Pauli 总能量。测量摘要的 `energy_per_site` 已除以格点数，再除以 4 即得论文口径的每格点能量。[Heisenberg 实现](../quantax/operator/common_operators.py) · [测量摘要示例](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/summary.json) · [论文与复现结果](reproduction-results.md#论文结果和模型配置)

`optimizer.VarE` 是局域能量的中心二阶矩估计。`H.expectation(state, samples, return_var=True)` 的第二个返回值在当前实现中是 $\langle|E_{\mathrm{loc}}|^2\rangle$，尚未减去均值平方；[`measure.py`](../examples/measure.py) 因此在汇总后另算方差。它们都不等同于能量均值的统计误差。

### 零方差外推

哈密顿量本征态的能量方差 $\sigma^2=\langle H^2\rangle-\langle H\rangle^2$ 为零。若几组变分态都接近同一个目标本征态，且误差成分的组成相近，能量偏差近似正比于方差。论文用 $\sigma^2/N$ 作横轴、$E/N$ 作纵轴拟合直线，零方差处的截距给出目标对称性下最低能量的估计。一步 Lanczos 在变分态及哈密顿量作用于该态所得向量张成的子空间里重新最小化能量，可以提供一个额外的拟合点。[论文式 (30)–(34)](Chen-2024-MinSR/paper.pdf#page=9) · [10×10 方格的拟合示例](reproduction-results.md#零方差拟合数据)

### 运行限制

晶格和部分对称性对象依赖全局状态并带有缓存。切换到另一教程、另一晶格尺寸或另一数值类型配置时，重启 Notebook 内核再执行。`max_parallel` 限制每台设备的前向批量，对称投影会占用该批量；它不直接限制所有导数和求解矩阵的内存。降低它可以缓解部分内存压力，样本数和网络规模也需配合调整。

JAX 环境分别位于 `environments/py39-jax0425/.venv/` 和 `environments/py311-jax062/.venv/`。环境选择脚本可按显卡同步对应的 JAX 项目。QuSpin wheel 需按 Python ABI 和目标系统构建。[安装说明](../README.md#installation) · [uv_machine.py](../scripts/uv_machine.py)
