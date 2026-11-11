# MinSR 项目分析

## 项目做了什么

MinSR 是论文 *Empowering deep neural quantum states through efficient optimization* 的配套仓库。它用神经量子态（Neural Quantum State，NQS）近似量子多体波函数，并提供一套从晶格建模、对称性约束、蒙特卡洛采样到参数优化和能量测量的实现。项目背景和论文入口见根目录的 [README](../README.md)。

这里的神经网络接收一个自旋构型，输出该构型的波函数振幅。训练目标是降低给定哈密顿量的变分能量。代码围绕自旋晶格模型展开，教程覆盖横场 Ising 模型、方格 Heisenberg 模型，以及方格和三角格上的受挫 $J_1$-$J_2$ 模型。

仓库名称是 MinSR，实际导入的 Python 包名是 `quantax`。MinSR 是其中的优化算法；Quantax 还负责晶格、算符、量子态、网络、对称性和采样。根目录说明将其定位为论文所用的历史版本源码，使用和修改时应以本仓库接口为准。

## 计算怎样完成

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

## MinSR 改进了哪一步

随机重构（Stochastic Reconfiguration，SR）根据波函数在参数空间中的几何结构选择更新方向。本仓库的 `TDVP` 默认执行虚时间优化，用来降低能量；`TDVP_exact` 用完整基底上的波函数计算对应更新，适合小系统检查。

用 $N_s$ 表示样本数，$N_p$ 表示可训练参数数。对于默认的非重加权、实参数和实输出情形，可把每个样本上的对数导数及局域能量中心化，写成矩阵和向量：

$$
O_{sk}=\frac{1}{\psi_\theta(s)}\frac{\partial\psi_\theta(s)}{\partial\theta_k},\qquad \bar O_{sk}=\frac{O_{sk}-\langle O_k\rangle}{\sqrt{N_s}},\qquad \bar\epsilon_s=\frac{E_{\mathrm{loc}}(s)-E}{\sqrt{N_s}}.
$$

参数空间中的 SR 方程涉及 $S=\bar O^\dagger\bar O$，其大小为 $N_p\times N_p$。当参数远多于样本时，MinSR 转而构造样本空间矩阵：

$$
T=\bar O\bar O^\dagger,\qquad \delta\theta=\bar O^\dagger T^+\bar\epsilon.
$$

这里的 $T^+$ 表示伪逆形式的求解；实现通过特征值分解和对小特征值的平滑抑制控制数值不稳定。实际调用 `state.update(eta * step)` 后，网络参数沿 $-\eta\delta\theta$ 更新，另外还处理整体尺度项。

这样，主要的矩阵求解从参数数决定的维度转到样本数决定的维度。当 $N_p\gg N_s$ 时，这一转换适合深网络。它仍需构造样本空间矩阵，样本数增大时也会增加内存和求解开销，不能仅靠增加样本数消除训练成本。

两种优化接口在本仓库中还有实现上的区别：

| 接口 | 导数与矩阵的处理 | 使用位置 |
| --- | --- | --- |
| `TDVP` | 先调用 `state.jacobian` 构造完整导数矩阵；默认求解器按矩阵形状选择样本空间或参数空间算法 | 5 个教程 |
| `MinSR` | 对支持的顺序网络分块反向计算导数，累加 $T$；求解后再计算导数共轭转置与向量的乘积，避免一次保存整个导数矩阵 | `examples/train.py`、`examples/train_symm.py` |
| `TDVP_exact` | 在指定对称性基底上计算完整波函数及导数，不依赖蒙特卡洛样本 | 教程 1 |

因此，教程中的 `TDVP` 在参数多于样本时也可能使用最小范数求解。`MinSR` 类进一步改变了导数的计算和保存方式。相关源码见 [tdvp.py](../quantax/optimizer/tdvp.py)、[minsr.py](../quantax/optimizer/minsr.py) 和 [solver.py](../quantax/optimizer/solver.py)。

上述矩阵尺寸说明针对实数情形。实参数、复输出时，实现会拼接实部和虚部，样本空间维度变为 $2N_s$。`MinSR` 明确拒绝非全纯复参数态（`vs_type == 1`），对实参数、复输出的网络也要求复数中间量只出现在末尾的无参数层中。

## 网络、对称性和采样各自负责什么

### 网络表示振幅和符号

教程 1 使用全连接受限玻尔兹曼机 `RBM_Dense`，其余教程使用残差卷积网络 `ResSum`。`ResSum(depth, channels, kernel_size)` 的前三个参数分别指定卷积层数、通道数和卷积核大小，`depth` 必须是偶数。卷积采用周期填充，末端通过空间求和或带平移特征标的求和生成波函数振幅。

默认实数设置下，`ResSum` 的指数输出给出正振幅。`use_sinh=True` 将末端激活改为带整体尺度的 $\sinh(x)+1$，使实输出可以变号。三角格教程则将默认计算类型设为复数，网络把实通道配成复数后输出复振幅；网络参数仍为实数。实现见 [sum_nets.py](../quantax/nn/sum_nets.py) 和 [activation.py](../quantax/nn/activation.py)。

### 对称性限制搜索空间

`TotalSz()` 指定总磁化为零的子空间，`SpinInverse()` 指定自旋反演偶扇区，`SpinInverse(-1)` 指定奇扇区。`Trans1D`、`Trans2D` 指定平移动量扇区，`C4v` 和 `D6` 分别用于方格和三角格点群。`+` 用来组合这些约束。

空间对称性和自旋反演通过对变换后的构型求值并加权平均来投影波函数。固定总磁化则还依赖基底和采样器：`TotalSz` 决定精确基底中的上自旋数，交换采样在该磁化扇区内移动。默认零动量的 `ResSum` 已通过周期卷积和末端求和编码平移对称性，因此教程 2、3 在变分态外部只增加自旋反演和点群投影。实现见 [symmetry.py](../quantax/symmetry/symmetry.py) 和 [nqs_layers.py](../quantax/nn/nqs_layers.py)。

### 采样器决定怎样探索构型

Ising 模型含单自旋翻转项，教程使用 `LocalFlip`。Heisenberg 类模型的教程固定总磁化，使用交换两个格点自旋的 `NeighborExchange`。参数 `n_neighbor=[1, 2]` 允许在第一、第二近邻键上提出交换，不会改变哈密顿量本身。

默认 `reweight=2.0` 按 $|\psi|^2$ 采样。大规模训练示例改用 `reweight=1.0`，按 $|\psi|$ 采样，并通过 `Samples.reweight_factor` 把统计量重加权回目标分布。训练脚本的注释标明，该设置仅用于不超过 100 个格点的系统。实现见 [metropolis.py](../quantax/sampler/metropolis.py) 和 [status.py](../quantax/sampler/status.py)。

## 代码和材料放在哪里

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
| [examples/](../examples/) | 较大系统的训练、对称化续训和权重测量脚本 |
| [params/](../params/) | 10×10 方格、$J_2/J_1=0.5$ 的 3 组已训练权重 |
| [data/](../data/) | `Fig2a.csv`、`Fig2b.csv`、`Fig2c.csv`、`Fig3.csv` 图表数据 |
| [images/](../images/) | 论文相关示意图和结果图 |

JAX 负责自动微分、编译和设备计算，Equinox 负责网络模块和参数序列化，QuSpin 负责自旋基底、算符作用和精确对角化。仓库支持同一节点上的多设备计算；根目录说明仍将多节点兼容列为待实现内容。

## 从教程到论文规模示例

[train.py](../examples/train.py) 在 10×10 周期方格上训练 $J_1=1$、$J_2=0.5$ 的模型，使用 `ResSum(16, 16, 3, use_sinh=True)`、10000 个样本和 `MinSR`，迭代 20000 次。每 10 次迭代保存能量和方差曲线，每 100 次保存一次权重。

[train_symm.py](../examples/train_symm.py) 读取 `params_20000.eqx`，加入 `TotalSz() + SpinInverse() + C4v()` 约束后继续训练 10000 次。该权重路径相对于工作目录；需要让它指向第一阶段生成的文件。两个训练脚本会使用相同的 `energy.pdf` 和 `VarE.pdf` 文件名，运行时注意输出目录。

[measure.py](../examples/measure.py) 读取仓库提供的权重并反复采样测量能量，不更新参数。修改脚本中的 `network_idx` 可选择对应网络：

| `network_idx` | 网络配置 | 参数文件 |
| --- | --- | --- |
| 1 | 16 层、16 通道、3×3 卷积核 | `square10x10_1616.eqx` |
| 2 | 16 层、32 通道、3×3 卷积核 | `square10x10_1632.eqx` |
| 3 | 30 层、64 通道、3×3 卷积核 | `square10x10_3064.eqx` |

测量脚本的注释将这些网络对应到论文图 2b 中的 ResNet2（MinSR）结果。`.eqx` 文件保存网络权重，加载时仍须创建匹配的网络结构。

## 读数和运行时需要核对的约定

`Heisenberg` 使用 Pauli 矩阵归一化，未启用 `msr` 时每条键对应 $J_{ij}\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_j$。对于通常的自旋算符 $\boldsymbol S=\boldsymbol\sigma/2$，相同耦合常数下能量相差 4 倍。`msr=True` 在第一近邻键上改变横向交换项的符号，纵向项和第二近邻项保持原样。比较教程、脚本与论文表格时，需要核对这一基底约定、能量尺度以及总能量和每格点能量的区别。源码见 [common_operators.py](../quantax/operator/common_operators.py)。

`optimizer.VarE` 是局域能量的中心二阶矩估计。`H.expectation(state, samples, return_var=True)` 的第二个返回值在当前实现中是 $\langle|E_{\mathrm{loc}}|^2\rangle$，尚未减去均值平方；`measure.py` 因此在汇总后另算方差。它们都不等同于能量均值的统计误差。

晶格和部分对称性对象依赖全局状态并带有缓存。切换到另一教程、另一晶格尺寸或另一数值类型配置时，重启 Notebook 内核再执行。`max_parallel` 限制每台设备的前向批量，对称投影会占用该批量；它不直接限制所有导数和求解矩阵的内存。降低它可以缓解部分内存压力，样本数和网络规模也需配合调整。

环境安装沿用根目录的 [安装说明](../README.md#installation)：当前依赖配置面向 Linux x86_64、Python 3.9 和 JAX 0.4.25，并引用相邻目录构建的历史 QuSpin wheel。硬件适配、CPU 执行和 CUDA 检查都已在该说明中列出。
