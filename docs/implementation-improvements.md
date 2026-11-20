# MinSR 实现改进记录

## Heisenberg 局域能量留在 GPU 上计算

### 原来的数据流

普通训练期间曾同时编译 QuSpin，步耗时从约 2.6 秒升到 13–14 秒，停止编译后恢复到约 2.7 秒。排查这次降速时，发现训练没有切到 CPU，但每步计算局域能量都要把数据在 CPU 和 GPU 之间来回搬运，GPU 前向计算还需要等 CPU 枚举完自旋构型。

具体来说，蒙特卡洛采样本来就在 GPU 上。算局域能量时，通用算符路径 [`Operator.psiOloc()`](../quantax/operator/operator.py#L247) 先用 `np.asarray(samples.spins)` 把样本拉回 CPU。QuSpin 在 CPU 上求对角矩阵元、枚举非对角项连接的自旋构型，NumPy 再把这些构型整理成网络的输入。网络在 GPU 上给出波函数值，程序按原样本汇总各个构型的贡献。

### 新路径怎么计算

[`Heisenberg()`](../quantax/operator/common_operators.py) 新增 `device_local` 选项，普通训练、对称续训和测量三个脚本都启用了它。构造算符时，`np.asarray(pairs)` 仍会生成主机端键表，用于保留原来的算符表示；键编号和耦合系数随后转成 JAX 数组。这一步只做一次，之后每步直接复用设备端数组。

1. `_heisenberg_local()` 读取每条键两端的自旋。所有键都贡献对角项 $J_{ij}s_i s_j$；只有自旋反向且横向系数通过原有的近零过滤时，才生成交换后的构型。横向系数取 $2J_{ij}$；开启 `msr` 时，只有第一近邻的横向项反号。
2. JAX 在设备上把有效交换的索引排在一起，用 `jax.lax.while_loop` 分批构造交换后的构型并计算波函数。每批构型数为 `MAX_PARALLEL // state.nsymm`，因此对称投影后的前向规模仍受 `MAX_PARALLEL` 限制。末批无效位置的能量贡献置零，不用把全部交换构型一次展开。
3. 按原样本累加所有贡献，得到 $(H\psi)(s)$；原有的 `Oloc()` 仍负责除以 $\psi(s)$。采样规则、对称投影和 MinSR 求解都没变。

[`Variational`](../quantax/state/variational.py) 把批量前向提取成 `_forward_batch`，供局域能量内核直接调用。新路径仍用全向上的自旋构型填充批次，延续原来计算最大波函数幅值的方式；这个幅值用于重缩放。记录最大幅值时用 `jnp.where` 比较两个设备标量，不再让 Python 先读取比较结果。训练循环里用于日志、绘图和计时的同步仍然存在。

新路径适用于单设备上的 `Variational` 自旋态，要求设置 `max_parallel`。没有开启 `device_local` 或不满足这些条件时，仍走通用 QuSpin 路径；精确对角化接口不受影响。

### 数值对照和速度

[`scripts/check_gpu.py`](../scripts/check_gpu.py) 在 4×4 方格上用相同样本对照两条路径，检查开启和关闭 Marshall 符号变换时的逐样本局域能量、期望值和未中心化二阶矩，也检查了没有可交换键的全极化构型。CPU 和 CUDA 检查均通过。再用已训练的 16 层、32 通道权重，在 10×10 方格上对同一批 1000 个样本复查，对称投影后的局域能量最大绝对差约为 $7\times10^{-10}$。

普通训练计时用同一张 RTX 4070 Ti 和同一个随机种子 `42`，分别启用和关闭 `device_local`，各从头跑 40 步。两次都使用 10×10 方格、$J_1=1$、$J_2=0.5$，以及 16 层、32 通道、3×3 卷积核的实值 ResNet2（`use_sinh=True`）。训练配置为 1000 个样本、`reweight=1.0`、`MAX_PARALLEL=32768` 和 `STEP_SIZE=0.001`。表中的单步耗时取第 11–40 步平均值，避开启动和首次编译。

| 局域能量路径 | 平均每步（秒） | 第 40 步能量 | 第 40 步方差 |
| --- | ---: | ---: | ---: |
| NumPy / QuSpin | 2.709 | -146.97308136 | 374.1553 |
| GPU | 2.472 | -146.97308136 | 374.1553 |

这次普通训练的平均步耗时减少了约 8.8%，40 步的能量和方差在日志精度内逐步一致。这个比例只来自普通训练，不能用作对称续训或测量的加速比。

复查命令：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m scripts.check_gpu
JAX_PLATFORMS=cuda .venv/bin/python -m scripts.check_gpu
```

### 对称续训的显存问题

32 通道网络的对称续训出现过两类显存消息。两条都写了 `ran out of memory`，但要看后续日志和训练是否继续，不能只看申请了多少显存。

| 日志特征 | 这次的结果 | 怎么处理 |
| --- | --- | --- |
| 申请 8.52 GiB 失败，日志注明 `this is not a failure` | XLA 试选卷积算法时，一次候选分配没有成功；验证仍显示 `Passed`，正式续训也继续记录步数。 | 只出现这条警告且训练仍在推进时，不用停跑。 |
| 申请约 8.1 GiB 失败，随后抛出 `RESOURCE_EXHAUSTED` | MinSR 反传播执行时申请不到所需的临时显存，这次验证未能完成更新。 | 不能忽略。调整显存分配后，先验证一次完整更新再重跑。 |

这次开启 `XLA_PYTHON_CLIENT_PREALLOCATE=true`，保持 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`，原配置通过了一次完整的 CUDA 更新，随后启动了正式续训。样本数、前向批量和训练步数都没改。预分配后仍出现过 8.52 GiB 的警告，但没有中断续训。`MAX_PARALLEL` 限制的是网络前向批量，不直接限制 MinSR 反传的显存。
