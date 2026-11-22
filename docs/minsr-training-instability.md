# MinSR 训练中能量异常和权重更新过大的排查

2026 年 11 月 21–22 日，RTX 5060 Ti 上的 PyTorch 移植验收遇到了两类异常。有些步的日志能量仍在正常范围，应用更新后波函数振幅却变成了 `Inf` 或 `NaN`。另一些普通训练运行在参数和振幅仍有限时，加权能量已经大幅偏离。排查以 10×10 周期方格、$J_1=1$、$J_2=0.5$、16 层 16 通道、1000 样本和固定步长 0.001 为主。能量均按 Pauli 哈密顿量的总能量口径记录。[历史验收与稳定性记录][history] · [当前验收配置](pytorch-migration-plan.md#完整验收结果)

## 从 Gram 矩阵报错追到上一轮更新

早期一轮普通训练跑满 2000 步，同一轮的最终权重进入对称性训练后，日志在第 98 步后报出 `linalg.eigh` 未收敛。普通训练能跑完，只说明这组参数在当时的普通训练中可以继续更新。对称性训练加入自旋反演和 $C_{4v}$ 投影，构型振幅、近邻振幅比和采样轨迹都随之改变。[普通训练日志][old-ordinary] · [最初的分解报错][first-eigh]

后来用这组普通训练最终权重从第 1 步重跑对称性训练，并在更新前保存权重、采样构型、振幅、局域能量、Gram 矩阵和求解右端。第 135 步的 Gram 矩阵和右端仍有限，求解也完成了，但实际施加的网络参数更新范数约为 22.10。更新前 1000 个测试构型的振幅都有限，更新后变成 0/1000。第 136 步才因为求解输入含非有限值而停止。最初的异常停在不同步数，未保存当步求解输入。第 135–136 步这次复现把大更新、振幅溢出和下一步的求解失败连在了一起。[对称性训练捕获日志][symm-capture] · [第 135 步分析][symm-analysis]

## 小振幅怎样影响一次 MinSR 更新

交换后的构型 $s'$ 与当前构型 $s$ 可以只差一条键上的自旋交换。局域能量把两者的振幅比按哈密顿量矩阵元加总。

$$
E_{mathrm{loc}}(s)=\frac{(H\psi)(s)}{\psi(s)}=H_{ss}+\sum_{s'\ne s} H_{ss'}\frac{\psi(s')}{\psi(s)}.
$$

即使分子很小，只要当前构型的振幅更接近零，比值就可能很大。对称性训练第 135 步的离群构型在普通训练最终权重下已有约 1752 的局域能量。到故障前，它的振幅约为 $-4.67\times10^{-26}$，$(H\psi)(s)$ 约为 $-1.26\times10^{-17}$，局域能量升到 $2.69\times10^8$。一条交换键贡献了分子的大部分，对应振幅比约为 $2.77\times10^8$。该构型对日志加权能量的贡献只有约 $7.6\times10^{-18}$，当步日志总能量仍约为 −199.17。[交换项分析][symm-outlier] · [不同权重上的同一构型][symm-start]

记 $N$ 为样本数，$w_i$ 为均值归一到 1 的重加权因子，$E_i$ 为局域能量，$O_i$ 为对数波函数对网络参数的导数。原版先以加权均值记录能量，再以算术平均值构造 MinSR 的右端和 Jacobian。

$$
E_{mathrm{log}}=\frac{1}{N}\sum_i w_i E_i,\qquad
\mu_E=\frac{1}{N}\sum_i E_i,\qquad
\mu_O=\frac{1}{N}\sum_i O_i,
$$

$$
b_i=\sqrt{\frac{w_i}{N}}(E_i-\mu_E),\qquad
\bar O_i=\sqrt{\frac{w_i}{N}}(O_i-\mu_O),\qquad
T=\bar O\bar O^\mathrm{T},\qquad
\Delta\theta=-\eta\,\bar O^\mathrm{T}T_{\varepsilon}^{+}b .
$$

这里 $T_{\varepsilon}^{+}$ 表示实现中带软截断的特征值伪逆，$\Delta\theta$ 表示对实值网络参数实际施加的改变量。离群构型在 $E_{mathrm{log}}$ 里受小权重压低，仍会以 $E_i/N$ 的幅度改变 $\mu_E$，进而同时移动整批样本的中心。第 135 步这一项把算术均值推到约 $2.69\times10^5$。固定 Gram 矩阵，单独保留均值偏移得到的更新范数约为 22.10，去掉这一偏移后约为 0.060。[JAX 中心化](../quantax/optimizer/tdvp.py) · [JAX MinSR](../quantax/optimizer/minsr.py) · [PyTorch 中心化](../quantax_torch/optimizer/tdvp.py) · [第 135 步右端分解][symm-analysis]

波函数的 `SinhShift` 包含 $\exp(x+\theta_0)$ 和 $\exp(-x+\theta_0)$。大更新能让网络输出 $x$ 和整体振幅偏移 $\theta_0$ 大幅变化。第 135 步复查中，最大的指数参数从约 4.64 升到 $1.83\times10^7$，振幅计算随即溢出。因此排查时要分别检查参数数组、实际施加的参数改变量和更新后的波函数振幅。[波函数激活实现](../quantax/nn/activation.py) · [更新前后的指数与振幅][symm-analysis]

## 普通训练中的两种异常

**日志能量看起来正常，更新后振幅溢出。** 在一次从 PyTorch 原生初值开始的复现中，第 649 步一条链上的构型振幅约为 $1.61\times10^{-9}$，局域能量约为 $5.06\times10^4$。第 650 步捕获到的同一采样行振幅降到约 $-9.79\times10^{-16}$，局域能量约为 $-5.96\times10^6$。其加权能量贡献只有约 $-5.0\times10^{-9}$，日志能量约为 −193.91，未加权均值却降至约 −6160。实际施加的参数更新范数约为 50.35，超过更新前参数向量的范数 23.17。固定 Gram 矩阵，去掉右端的均值偏移后，更新范数约为 0.066。应用原更新后，34944 个可训练参数都有限，复查的 1000 个振幅全部非有限。用保存的权重和构型把这一步的网络计算改成 FP64 复算，结果仍然如此。[第 649–650 步右端分析][ordinary-decompose] · [参数更新分布][ordinary-update] · [FP64 固定输入复算][ordinary-fp64]

**加权能量已偏离正常范围，数值仍有限。** 另一次普通训练先跑到第 700 步，再冻结权重检查。用原交换规则从随机、Néel 和条纹构型开始，得到明显不同的能量估计。例如热化 2000 次提议后，随机和条纹起点分别约为 +97 和 +116，Néel 起点约为 −196。随后从同一检查点做诊断更新时，一个记录批次的加权能量为 +102.77，有效样本数约为 4.93。这里的有效样本数按 $p_i=w_i/N$ 和 $N_{mathrm{eff}}=1/\sum_i p_i^2$ 计算。该批次的一行在伪逆中造成约 2.09 的参数更新范数，去掉它对应的右端分量后约为 0.052。这个案例同时显示了权重过度集中、采样依赖起点和伪逆放大，单看 `isfinite` 无法识别这类异常。[冻结权重采样对照][frozen-chains] · [第 17 个诊断步分析][ordinary-finite]

## 怎样排查移植、设备优化和初值

复算时固定故障前权重与采样构型，再比较局域能量、右端、Gram 矩阵、参数更新和更新后的振幅。固定输入后，两次故障都能在 JAX CPU 上复现。此前的性能优化仍可能因浮点运算顺序改变长期轨迹，复算表明触发这两次异常无需依赖 PyTorch 专用 CUDA 内核。[对称性训练 JAX CPU 复算][symm-jax] · [普通训练 JAX CPU 复算][ordinary-jax] · [普通训练 PyTorch 通用路径复算][ordinary-fixed]

| 固定输入 | PyTorch 更新范数 | JAX CPU 更新范数 | 两边更新后有限振幅 |
| --- | ---: | ---: | ---: |
| 对称性训练第 135 步 | 22.10256 | 22.10215 | 0/1000 |
| 普通训练第 650 步 | 50.34948 | 50.34937 | 0/1000 |

另一组实验用同一份保存的 PyTorch 第 0 步权重、同一块 RTX 5060 Ti 分别重跑优化路径和通用路径。两条路径都复现了普通训练异常，最后记录到第 1134 步和第 759 步。故障会随轨迹出现在不同步数。RTX 4070 Ti 此前两次完整验收都通过，且早期 RTX 5060 Ti 也有一轮普通训练跑满 2000 步。两次 RTX 4070 Ti 验收没有逐项保存第 0 步权重，无法把跨卡结果差异归到某一个因素。[同一初值的优化路径][optimized-replay] · [同一初值的通用路径][generic-replay] · [历史三组验收][history]

种子 2347 在 JAX 和 PyTorch 中对应不同的随机数生成器和初始化实现。实际保存的两组第 0 步权重有 34944 个可训练分量，只有 240 个完全相等，向量余弦约为 0.00044。用同一批初始构型测试，两组的初始能量都约为 −102.9，第一次更新也都有限。PyTorch 网络参数在 CPU 上初始化后迁往设备，初始自旋和后续采样使用训练设备的随机数生成器。故障构型何时被采到取决于训练轨迹和采样状态，复算时需要保存具体权重和构型。[初始权重逐项比较][init-weights] · [同构型的初次更新][init-samples] · [PyTorch 随机数生成器](../quantax_torch/global_defs.py)

## 试过的处理与最终取舍

| 处理 | 固定输入或短跑观察 | 最终取舍 |
| --- | --- | --- |
| 对过大的更新按比例限幅 | 一轮对称性训练触发 13 次。第 359 步将原更新乘以 0.797062。 | 更新量随每步数据改变，撤回。 |
| 改用加权均值中心化 | 对称性训练第 135 步的更新范数从约 22.10 降到 0.060，更新后的振幅仍有限。 | 改变了原版求解右端，作为归因实验保留，生产代码恢复原式。 |
| 缩小固定步长 | 同一故障输入上，步长 0.0005 时 1000 个振幅仍全部溢出，0.0001 时全部有限。 | 单步对照用来确认更新幅度的影响，验收使用原步长 0.001。 |
| 增加样本 | 第 650 步保留原来 1000 个样本，再加入 3000 个新样本，更新范数反而由 50.35 增至 182.19。 | 批次的均值和 Gram 矩阵同时改变，该对照只说明增样本未消除这次失稳。 |
| 并行回火 | 冻结权重后，让按不同振幅幂次采样的链交换构型。一次对照中，随机和 Néel 起点的能量都在 −197 附近。 | 这会改变采样转移过程，实验代码从主流程剔除。 |

[限幅试跑日志][clipped-run] · [加权中心化对照][symm-analysis] · [固定步长对照][reduced-step] · [增加样本对照][more-samples] · [冻结权重的回火对照][tempering]

最终代码保留原版的中心化、软截断伪逆、固定步长和邻居交换采样。PyTorch 的默认 FP32 参数更新改为先把改变量舍入到参数精度，再应用到权重，顺序与 JAX 一致。求解前检查 Gram 矩阵和右端是否有限，训练入口也检查更新是否有限，发现异常时停止。同卡一轮原生 JAX 已用种子 2347 跑完 2000 步普通训练，随后验收改用保存的 JAX CUDA 第 0 步权重启动 PyTorch 全流程。[同卡 JAX 普通训练][native-jax] · [验收提交与代码差异][acceptance-commit] · [参数更新实现](../quantax_torch/state/variational.py) · [求解器实现](../quantax_torch/optimizer/solver.py)

## 完整验收与复跑

RTX 5060 Ti 的验收使用 JAX 在 CUDA 上以种子 2347 生成的第 0 步权重。PyTorch 从第 1 步跑完全部 2000 步普通训练，对称性训练读取这一轮的最终权重并从第 1 步跑完 1000 步，最后固定本轮对称权重测量 100 轮。训练统计取末 200 步，测量取 100 轮的汇总。按移植文档统一容差，三个阶段均通过。[完整验收标准](pytorch-migration-plan.md#完整验收结果)

| 阶段 | Pauli 总能量 | 中心方差 |
| --- | ---: | ---: |
| 普通训练 | −197.76580 | 18.77921 |
| 对称性训练 | −198.81113 | 6.72867 |
| 固定权重测量 | −198.83452640 | 4.75737444 |

[普通训练日志][accepted-ordinary] · [对称性训练日志][accepted-symmetric] · [测量摘要][accepted-measurement] · [JAX 参考测量](../runs/j1j2_square_10x10_J1-1_J2-0.5/20261119_150209/summary.json)

将本次验收的最终对称权重套回旧故障的 1000 个构型后，原来那条约 $2.69\times10^8$ 的局域能量降到 −44.18，整批的参数更新范数约为 0.039，更新后的振幅 1000/1000 有限。这项复查用的是同一批旧构型，直接对照了两份权重对同一问题输入的响应。[最终权重对旧故障批次的复查][accepted-fault]

三个阶段的验收日志、最终参数和测量摘要已提交到 Git。故障快照和试跑日志保存在原工作区。两个快照都包含更新前的权重和该步的采样构型。[普通训练第 650 步快照][ordinary-snapshot] · [对称性训练第 135 步快照][symmetric-snapshot]

用本地保存的 [JAX 第 0 步权重][saved-init]，可以从仓库根目录通过 [PyTorch 流水线](../examples_torch/run_pipeline.py) 复跑。

```bash
initial_weights=runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_153201/params_0000.eqx
.venv/bin/python -m examples_torch.run_pipeline \
  --device cuda --seed 2347 --nsamples 1000 \
  --steps 2000 --symm-steps 1000 --measure-steps 100 \
  --initial-weights "$initial_weights"
```

[history]: https://github.com/GhostArtyom/MinSR/blob/ba2f4f8383db31fab70a4adf03a6d66072453d0a/docs/pytorch-migration-plan.md#完整验收结果
[acceptance-commit]: https://github.com/GhostArtyom/MinSR/commit/ba2f4f8383db31fab70a4adf03a6d66072453d0a
[accepted-ordinary]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_153201/training.log
[accepted-symmetric]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_160638/training_symm.log
[accepted-measurement]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_184508/summary.json
[saved-init]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261122_153201/params_0000.eqx
[ordinary-snapshot]: ../runs/torch-perf/eigh-diagnosis/random-init-2347-replay-20261122_081658/event_0650.pt
[symmetric-snapshot]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/event_0135.pt
[old-ordinary]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_021725/training.log
[first-eigh]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_025045/training_symm.log
[clipped-run]: ../runs/j1j2_square_10x10_J1-1_J2-0.5/20261121_124631/training_symm.log
[symm-capture]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/diagnostic.log
[symm-analysis]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/event_0135-analysis.log
[symm-outlier]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/event_0135-outlier.log
[symm-start]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/compare-starting-weights.log
[symm-jax]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/jax-replay-135/replay.log
[reduced-step]: ../runs/torch-perf/eigh-diagnosis/capture-gpu-20261121_185620/event_0135-fixed-step.log
[frozen-chains]: ../runs/torch-perf/eigh-diagnosis/replay-ordinary-from700-20261121_231707/frozen-chain.log
[ordinary-finite]: ../runs/torch-perf/eigh-diagnosis/replay-ordinary-from700-20261121_231707/event_0017-analysis.log
[ordinary-decompose]: ../runs/torch-perf/eigh-diagnosis/random-init-2347-replay-20261122_081658/decompose-649-650.log
[ordinary-update]: ../runs/torch-perf/eigh-diagnosis/fixed-torch-init-2347-20261122_180719/update-distribution.log
[ordinary-fp64]: ../runs/torch-perf/eigh-diagnosis/random-init-2347-replay-20261122_081658/fp64-step-649-650.log
[ordinary-jax]: ../runs/torch-perf/eigh-diagnosis/random-init-2347-replay-20261122_081658/jax-step650/replay.log
[ordinary-fixed]: ../runs/torch-perf/eigh-diagnosis/random-init-2347-replay-20261122_081658/fixed-failure-650.log
[optimized-replay]: ../runs/torch-perf/eigh-diagnosis/fixed-torch-init-2347-20261122_180719/optimized-1/j1j2_square_10x10_J1-1_J2-0.5/20261122_192533/training.log
[generic-replay]: ../runs/torch-perf/eigh-diagnosis/fixed-torch-init-2347-20261122_180719/generic-1/j1j2_square_10x10_J1-1_J2-0.5/20261122_194443/training.log
[init-weights]: ../runs/torch-perf/eigh-diagnosis/torch-jax-cuda-init-2347-20261122_153200/initial-weight-check.log
[init-samples]: ../runs/torch-perf/eigh-diagnosis/fixed-torch-init-2347-20261122_180719/same-first-sample-initial-weights.log
[more-samples]: ../runs/torch-perf/eigh-diagnosis/fixed-torch-init-2347-20261122_180719/sample-count-independent-step0650.log
[tempering]: ../runs/torch-perf/eigh-diagnosis/capture-ordinary-tempered-20261121_232803/independent-beta2.log
[native-jax]: ../runs/torch-perf/eigh-diagnosis/native-jax-seed2347-20261122_085803/j1j2_square_10x10_J1-1_J2-0.5/20261122_085804/training.log
[accepted-fault]: ../runs/torch-perf/eigh-diagnosis/fixed-torch-init-2347-20261122_180719/final-symmetry-old-fault-full-batch.log
