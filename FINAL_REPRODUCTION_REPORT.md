# DEQ-MPI 论文理解与复现报告

## 1. 论文主要内容

这篇论文面向磁粒子成像（Magnetic Particle Imaging, MPI）重建问题。MPI 的测量过程通常是一个病态线性逆问题：系统矩阵维度高、噪声显著，直接用最小二乘、L1、TV 或 ADMM 类方法容易在细节、噪声抑制和结构一致性之间摇摆。

论文提出的 DEQ-MPI 把 MPI 重建写成一个“深度平衡”问题：不显式堆叠很多网络层，而是学习一个迭代算子，并把最终重建定义为该算子的固定点。模型核心由三部分组成：

1. 用截断 SVD/最小二乘结果作为物理一致的初始图像。
2. 用 RDN 图像先验网络学习血管图像空间的结构约束，起到去噪/投影作用。
3. 用 learned consistency 网络学习与系统矩阵和测量数据相关的数据一致性校正。

训练时通过 Anderson acceleration 求固定点，并用隐式微分进行反向传播；推理时迭代到固定点，得到最终重建。这样模型表现得像“无限层展开网络”，但不需要真的保存每一层中间激活。

## 2. 主要创新点

- 将深度平衡模型引入 MPI 重建，把传统迭代重建和深度先验统一到固定点框架中。
- 用隐式微分替代显式长链反向传播，降低深层迭代模型训练时的显存压力。
- 将物理前向模型、截断 SVD 初始化、图像先验和数据一致性网络组合起来，比纯图像后处理更贴近 MPI 逆问题本身。
- 分阶段训练设计清晰：先训练 RDN 图像先验，再训练 learned consistency，最后联合训练 DEQ-MPI。
- 在模拟数据和实验 phantom 上均展示了较强的噪声鲁棒性；在约 15 dB 测量 SNR 设置下，DEQ-MPI 明显优于 L1、TV、L1+TV 和 PP-MPI。

## 3. 可提升空间

- 训练流程可复现性还可以更强：原仓库没有完整保存 optimizer/scheduler 状态，长时间训练中断后不能做到 bit-exact resume。
- 代码中有少量兼容性和工程细节问题，例如现代 PyTorch 下 in-place autograd 报错、短训练时 StepLR step size 为 0、部分 checkpoint 保存条件优先级不清晰。
- paper setting 和代码默认值存在需要标注的差异：训练使用 Anderson acceleration；部分推理加载路径使用普通固定点迭代。
- 评估脚本可以更模块化。目前官方 `inferenceSimulated.py` 同时评估多个方法并保存较大 `.mat` 文件；单 checkpoint 快速评估需要额外脚本。
- 可以进一步补充跨数据集、真实实验数据、不同系统矩阵扰动、不同 SNR 下的泛化测试。
- 现有网络比较轻量，后续可尝试更强的图像先验、显式不确定性估计或物理约束更强的数据一致性模块。

## 4. 本地复现环境

- 系统：Windows / PowerShell
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB
- Python：3.10.20，虚拟环境 `.venv`
- PyTorch：2.11.0 + CUDA 12.8
- 数据：作者提供的 HDF5 simulated vessel 数据，共 3,730 张测试图像

原论文/仓库给出的 PyTorch 1.11 + CUDA 11.3 组合不支持本机 RTX 5060，因此本次使用 `requirements-repro-cu128.txt` 记录现代 CUDA 12.8 兼容环境，同时保留原 `requirements.txt` 不改动。

## 5. 本次工程修改

本次修改集中在“能稳定复现且不覆盖作者 checkpoint”：

- 给 `train_ppmpi.py`、`train_dcDenoiser.py`、`train_deqmpi.py` 增加 `--outputRoot`，所有自训练产物写入 `training/reproduction/`。
- 修复短 smoke test 下 `StepLR(epoch_nb // 5)` 可能为 0 的问题。
- 修复 learned-consistency checkpoint 条件中的优先级问题。
- 修复 modern PyTorch 下 consistency normalization 的 in-place autograd 问题。
- 给 DEQ-MPI 增加 `--resumeModel` 和 `--startEpoch`，支持长训练分段恢复。
- 将 DEQ-MPI 周期 checkpoint 改为 epoch 训练/验证结束后保存，避免 checkpoint 名称和实际权重进度错位。
- 新增 `eval_repro_deqmpi.py`，用于快速评估单个复现 checkpoint 并输出 JSON 摘要。

## 6. 复现步骤

环境安装：

```powershell
uv python install 3.10
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe -r requirements-repro-cu128.txt
```

官方 checkpoint 推理 sanity check：

```powershell
.venv\Scripts\python.exe inferenceExperimental.py
.venv\Scripts\python.exe inferenceSimulated.py
```

RDN 图像先验预训练：

```powershell
.venv\Scripts\python.exe train_ppmpi.py --outputRoot training/reproduction/denoiser
```

learned-consistency 预训练：

```powershell
.venv\Scripts\python.exe train_dcDenoiser.py --outputRoot training/reproduction/dcDenoiser
```

DEQ-MPI 联合训练：

```powershell
.venv\Scripts\python.exe train_deqmpi.py `
  --outputRoot training/reproduction/deqmpi `
  --preLoadDir training/reproduction/denoiser/ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_nF12_nB4_lieb4_gr12_rMn0.5_1.0/epoch200END.pth `
  --preLoadDirDC training/reproduction/dcDenoiser/dcDenoiserPsi_1D_ds_lr_0.001_wd_0_bs_64_pSNR_18.0_fixNs_1_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_sN_0.02_ls_0/epoch200END.pth
```

如果训练中断，可用完整 DEQ checkpoint 恢复，例如：

```powershell
.venv\Scripts\python.exe train_deqmpi.py `
  --outputRoot training/reproduction/deqmpi `
  --resumeModel training/reproduction/deqmpi/DeqMPI_1D_ds_lr_0.001_wd_0_bs_64_pSNR_10.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1/epoch150.pth `
  --startEpoch 150
```

最终 checkpoint 评估：

```powershell
.venv\Scripts\python.exe eval_repro_deqmpi.py `
  --checkpoint training/reproduction/deqmpi/DeqMPI_1D_ds_lr_0.001_wd_0_bs_64_pSNR_10.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1/epoch200END.pth `
  --outJson training/reproduction/deqmpi_eval_epoch200END.json
```

## 7. 最终复现结果

| 阶段 | 结果 |
|---|---:|
| RDN final train pSNR | 29.563950 dB |
| RDN final validation pSNR | 29.558994 dB |
| learned-consistency final train pSNR | 67.923256 dB |
| learned-consistency latest validation pSNR | 68.029541 dB |
| DEQ-MPI final train pSNR | 26.825563 dB |
| reproduced DEQ-MPI test mean pSNR | **32.193039 dB** |
| reproduced DEQ-MPI test std | 4.514024 dB |
| paper Table I DEQ-MPI at 15 dB | 32.1 ± 4.5 dB |

最终复现 checkpoint 在 3,730 张测试图像上的均值 `32.193039 dB`，与官方 checkpoint 推理结果 `32.19 dB` 以及论文 Table I 的 `32.1 ± 4.5 dB` 一致。

## 8. 关键产物

- RDN checkpoint SHA256:
  `BF63ABC68155BA39C40C4694D845C5C0171078BE45E83C1ADB11285622F5B92B`
- learned-consistency checkpoint SHA256:
  `B28549B0F23665187D6DD9926AA3AB71870925CA4FCB6B64073A66E9F8996183`
- DEQ-MPI checkpoint SHA256:
  `40FA594874FA008687E204A0CBB8661B24015205CBE7C440DD6493281EBED38C`
- final evaluation JSON SHA256:
  `2F07C380DBCBAAA7378A9C6027F3D8C9C4C5F1916E187FFF04FBC522D8A25F90`

详细数值见 `REPRO_RESULTS.md`，环境与补丁说明见 `REPRODUCTION.md`。
