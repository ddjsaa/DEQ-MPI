# ADMM-DEQ-MPI 中文执行摘要

版本: 0.2  
日期: 2026-07-05  
工作目录: `D:\VSCode\DEQ-MPI-repro`

这份中文文件是 `ADMM_DEQ_MPI_PROJECT_PLAN.md` 的可读执行摘要。英文文件保留完整细节；本文件用于后续协作时快速确认研究方向、已完成内容、证据边界和下一步。

## 1. 当前研究方向

项目面向磁粒子成像的重建问题，目标是在官方 DEQ-MPI 复现基础上，发展一个统一的 Sparse-PnP-ADMM-DEQ 重建框架。

核心思想是把三类约束放进同一个固定点模型:

- 物理数据一致性: 使用 MPI 系统矩阵和测量数据约束重建结果。
- Plug-and-Play 图像先验: 使用学习型去噪器提供图像结构先验。
- 显式稀疏先验: 使用非负 l1 proximal 分支控制稀疏性、尾部强度和支持区域。

当前最稳妥的论文主张不是“全面超过 DEQ-MPI”，而是:

> Sparse-PnP-ADMM-DEQ 在 paired canonical 2D benchmark 上达到 DEQ-MPI 级别的重建质量，同时提供更快的推理设置，并给出更可解释的 fixed-point / ADMM 结构。

## 2. 已完成工作

### 官方 DEQ-MPI 复现

官方 DEQ-MPI 已完成本地复现。

- reproduced DEQ-MPI test mean pSNR: `32.193039 dB`
- test std: `4.514024 dB`
- paper Table I reference: `32.1 +/- 4.5 dB`

主要记录文件:

- `REPRO_RESULTS.md`
- `REPRODUCTION.md`
- `FINAL_REPRODUCTION_REPORT.md`

### Sparse-PnP-ADMM-DEQ 实现

核心实现位于:

- `sparse_pnp_admm_deq.py`

已实现内容:

- fixed-point state: `w = (x, z, s, u, v)`
- data-consistency x-update
- denoiser branch
- non-negative l1 prox branch
- dual updates
- fixed-point residual diagnostics
- finite-iteration inference
- DEQ wrapper with Anderson / fixed solver support

主要训练和评估脚本:

- `train_sparse_pnp_admm_deq.py`
- `eval_sparse_pnp_admm_deq.py`
- `eval_sparse_pnp_admm_deq_checkpoint.py`
- `eval_paired_sparse_pnp_deqmpi.py`
- `eval_sparse_pnp_deq_convergence_trace.py`

### 已有证据包

已有结果包括:

- full-test paired evaluation
- quality/runtime figure
- minimal ablation
- pSNR / SVD rank / direct-vs-canonical stress tests
- convergence traces
- OpenMPI 3D feasibility validation
- manuscript-style draft

关键数字:

- quality setting mean delta vs DEQ-MPI: `+0.0018 dB`
- fast setting mean delta vs DEQ-MPI: `-0.0554 dB`
- fast setting speedup: about `3.18x`
- quality final mean solver residual: `0.012652`
- fast final mean solver residual: `0.023828`

## 3. 不能过度声称的内容

后续写论文和汇报时必须避免以下表述:

- 不要声称 Sparse-PnP-DEQ 全面优于 DEQ-MPI。
- 不要声称 l1 分支在当前 benchmark 上显著提升 mean PSNR。
- 不要把 OpenMPI 3D 当作有 ground truth 的定量 benchmark。
- 不要声称所有 pSNR 下都有强鲁棒性；当前 pSNR 8 和 15 有明显 gap。
- 不要声称 DEQ solver 在 PSNR 上总是优于 finite unrolling。

当前支持的说法是:

- 主 paired canonical 设置下质量与 DEQ-MPI 基本持平。
- 推理速度有优势。
- denoiser branch 是关键模块。
- supervised Sparse-PnP tuning 有明显贡献。
- SVD rank 180/220/250 范围内结果稳定。
- OpenMPI 3D 结果可以作为真实数据可行性证据。

## 4. 后续优先级

### 第一优先级: 整理提交和仓库结构

已开始按功能拆分 commit:

1. reproducible DEQ-MPI training workflow
2. Sparse-PnP-ADMM-DEQ method tooling
3. zero-shot and OpenMPI evaluation tools
4. manuscript and planning documents

不要把以下内容提交进普通 git:

- `.venv/`
- `datasets/`
- `training/reproduction/`
- `external/DPIR/`
- `external/models/`
- `*.pth`, `*.pt`, `*.ckpt`
- 大型 figure exports 和 checkpoint artifacts

### 第二优先级: 收紧 manuscript claim

论文应围绕如下证据组织:

- paired evaluation 是主线
- quality-speed tradeoff 是主结果
- l1 分支作为可控稀疏/支持约束讨论，不作为 mean-PSNR 主贡献
- OpenMPI 作为 feasibility / transfer evidence
- pSNR extrapolation 作为 limitation

### 第三优先级: 可选补充实验

如果还要补实验，优先考虑:

- support ratio / tail intensity / p95-p99 等支持稀疏分支的指标
- pSNR 8 和 pSNR 15 的 noise-aware retuning
- finite vs DEQ solver 的更清晰对照
- 方法图和 equation-ready Methods section

## 5. 当前最合理的下一步

短期内不要再盲目跑大实验。建议顺序:

1. 完成剩余文档 commit。
2. 决定是否保留 regenerated `Experimental15dB.mat` 和 `experimentalOutput15dB.png`。
3. 处理只剩末尾换行变化的文件。
4. 把 manuscript draft 转成正式论文结构，补引用、公式和实验细节。
5. 如需增强稀疏性主张，再补 support/tail-intensity 指标，而不是单纯追求 PSNR。
