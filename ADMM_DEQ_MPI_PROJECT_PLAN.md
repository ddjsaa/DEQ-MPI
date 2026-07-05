# ADMM-DEQ-MPI Project Plan

Version: 0.1  
Date: 2026-07-04  
Workspace: `D:\VSCode\DEQ-MPI-repro`

This document is the execution guide for the MPI reconstruction project. Unless the user explicitly changes direction, all future coding, experiments, figures, and manuscript planning should follow this file.

## 1. Core Research Direction

The project will develop a magnetic particle imaging (MPI) reconstruction method that unifies:

- explicit sparse regularization through an \(l^1\) proximal branch;
- implicit image prior regularization through a 2D plug-and-play denoiser;
- physics/data consistency through an ADMM-style variable-splitting update;
- fixed-point training/inference through a Deep Equilibrium Model (DEQ).

The target manuscript direction is:

> Sparse-Regularized Plug-and-Play ADMM Deep Equilibrium Reconstruction for Magnetic Particle Imaging

The central technical claim should be:

> MPI reconstruction can be improved or made more controllable by embedding explicit sparsity and implicit denoising priors into a single ADMM fixed-point map, then solving and training this map as a DEQ rather than as a finite unrolled network.

## 2. Non-Negotiable Positioning

The existing repository already contains a strong DEQ-MPI reproduction and a ZeroShot-\(l^1\)-PnP extension. These must be positioned correctly.

### 2.1 What We Should Claim

We may claim that:

- DEQ-MPI is the strong supervised in-domain baseline.
- ZeroShot-PnP and ZeroShot-\(l^1\)-PnP are useful low-training or real-data transfer branches.
- The new contribution is the integration of the \(l^1\) proximal branch directly inside the ADMM-DEQ fixed-point state.
- OpenMPI results provide realistic phantom evidence, especially for qualitative robustness and transfer behavior.

### 2.2 What We Must Not Claim

We must not claim that:

- the current ZeroShot-\(l^1\)-PnP script alone is the final proposed method;
- simply adding \(l^1\) after DEQ-MPI is the main novelty;
- the current OpenMPI experiments are a full 3D supervised DEQ-MPI comparison;
- the method is universally better unless the experiments support that statement;
- \(l^1\) always improves PSNR, since current 2D evidence suggests it can reduce PSNR on dense vessel-like targets.

If a future result contradicts this plan, revise the claim before writing the manuscript.

## 3. Existing Repository Assets

### 3.1 Main Code Assets

The following files are relevant and should be reused where possible:

- `README.md`: official DEQ-MPI repository description.
- `modelClasses.py`: DEQ-MPI, fixed-point modules, Anderson/regular fixed-point solvers, and implicit gradient wrapper.
- `train_deqmpi.py`: supervised DEQ-MPI and ADMLD training entry point.
- `trainerClasses.py`: training loops, noisy data generation, and SVD initialization utilities.
- `data.py`: HDF5 patch loader and MPI system matrix loader.
- `reconAlgos.py`: classical ADMM reconstruction utilities.
- `reconUtils.py`: ART, PnP wrappers, TV prox, \(l^1\) soft-threshold, nonnegative soft-threshold, and projection operators.
- `zeroshot_l1_pnp.py`: existing ZeroShot-PnP and ZeroShot-\(l^1\)-PnP implementation.
- `make_2d_paper_comparison.py`: 2D benchmark construction and figure export.
- `eval_noise_sweep.py`: low-SNR evaluation framework.
- `eval_deq_initialized_zeroshot.py`: DEQ-initialized zero-shot refinement evaluation.
- `openmpi3d_tools.py`: OpenMPI MDF inspection/conversion into real-valued linear systems.
- `eval_zeroshot_openmpi3d.py`: OpenMPI 3D zero-shot reconstruction.
- `make_openmpi_paper_figures.py`: OpenMPI figure export.

### 3.2 Main Data Assets

Current data assets:

- `datasets/trainPatches.h5`: 33,692 training images, each `32 x 32`.
- `datasets/valPatches.h5`: 3,377 validation images, each `32 x 32`.
- `datasets/testPatches.h5`: 3,730 test images, each `32 x 32`.
- `inhouseData/expMatinHouse.mat`: in-house system matrix, `Aconcat` shape `(5486, 338)`.
- `inhouseData/expPhantominHouse.mat`: in-house experimental phantom data.
- `inhouseData/selElems.mat`: selected measurement elements.
- `datasets/OpenMPI/*.mdf`: public OpenMPI calibration and phantom measurements.
- `datasets/OpenMPI/*_real_system.npz`: converted OpenMPI linear systems.
- `external/models/drunet_gray.pth`: DPIR DRUNet grayscale denoiser weights.

### 3.3 Existing Reproduction Results

Existing local results indicate:

- LS/SVD on 2D simulated benchmark: about `12.3 dB`.
- supervised DEQ-MPI on 2D simulated benchmark: about `32.2 dB`.
- ZeroShot-PnP on 2D simulated benchmark: about `29.3 dB`.
- ZeroShot-\(l^1\)-PnP on 2D simulated benchmark: about `28.7 dB`.
- DEQ-initialized one-step ZeroShot-PnP: about `31.6 dB`.

Interpretation:

- DEQ-MPI remains the strongest current in-domain baseline.
- ZeroShot branches are useful but should not be presented as superior to DEQ-MPI on the matched supervised 2D setting.
- The new method must improve the fixed-point formulation itself, not merely post-process DEQ-MPI outputs.

## 4. Proposed Method

### 4.1 Optimization View

The intended reconstruction objective is:

\[
\min_x \frac{1}{2}\|Ax-y\|_2^2 + \lambda\|\Psi x\|_1 + R_D(x), \quad x \ge 0
\]

where:

- \(A\) is the MPI system matrix;
- \(y\) is the measured MPI signal;
- \(x\) is the tracer concentration image;
- \(\Psi\) is the sparsifying transform;
- \(R_D\) is an implicit prior represented by a 2D denoiser \(D_\theta\);
- \(x \ge 0\) enforces nonnegative tracer concentration.

First implementation:

- set \(\Psi = I\);
- apply \(l^1\) directly to the concentration image;
- use nonnegative soft-thresholding when appropriate.

Later extension:

- replace \(\Psi = I\) with wavelet or TV-like transforms only after the identity version is stable.

### 4.2 ADMM Fixed-Point State

The proposed fixed-point state should include:

\[
w=(x,z,s,u,v)
\]

where:

- \(x\): reconstructed MPI concentration;
- \(z\): denoiser/PnP branch variable;
- \(s\): sparse \(l^1\) branch variable;
- \(u\): dual variable for \(x=z\);
- \(v\): dual variable for \(x=s\).

The first target update is:

\[
x^{k+1} =
(A^TA+\rho I+\eta I)^{-1}
\left(A^Ty+\rho(z^k-u^k)+\eta(s^k-v^k)\right)
\]

\[
z^{k+1}=D_\theta(x^{k+1}+u^k)
\]

\[
s^{k+1}=\operatorname{soft}(x^{k+1}+v^k,\lambda/\eta)
\]

\[
u^{k+1}=u^k+x^{k+1}-z^{k+1}
\]

\[
v^{k+1}=v^k+x^{k+1}-s^{k+1}
\]

with optional nonnegativity:

\[
x,z,s \leftarrow \max(x,z,s,0)
\]

The DEQ formulation is:

\[
w^\ast=T_\theta(w^\ast;y,A)
\]

The model output is \(x^\ast\), the \(x\)-component of \(w^\ast\).

### 4.3 Implementation Name

Use one clear method name in code and manuscript:

`SparsePnPADMMDEQ`

Acceptable manuscript label:

`Sparse-PnP-ADMM-DEQ`

Do not introduce multiple competing names unless the method genuinely changes.

## 5. Coding Plan

### 5.1 File Organization

New code should be added without damaging the official reproduction path.

Recommended new files:

- `sparse_pnp_admm_deq.py`
  - new fixed-point module;
  - ADMM state packing/unpacking;
  - \(x,z,s,u,v\) updates;
  - optional nonnegative projection;
  - residual diagnostics.

- `train_sparse_pnp_admm_deq.py`
  - training entry point for the proposed method;
  - should reuse data loading and system matrix preparation from existing DEQ-MPI scripts.

- `eval_sparse_pnp_admm_deq.py`
  - test-set evaluation;
  - should output JSON and CSV metrics.

- `eval_sparse_pnp_admm_deq_sweeps.py`
  - noise, mismatch, undersampling, and hyperparameter sweeps.

- `make_sparse_pnp_admm_deq_figures.py`
  - manuscript figure export for the proposed method.

Recommended output directory:

- `training/reproduction/sparse_pnp_admm_deq/`

Do not overwrite:

- official checkpoints in `training/deqmpi/`;
- existing reproduction checkpoints unless explicitly requested;
- OpenMPI raw `.mdf` files.

### 5.2 Reuse Rules

Reuse existing utilities when possible:

- reuse HDF5 data loading from `data.py`;
- reuse SVD/system matrix preparation from `make_2d_paper_comparison.py`;
- reuse `softTpos` or equivalent nonnegative thresholding from `reconUtils.py`;
- reuse `DEQFixedPoint` from `modelClasses.py` if its interface remains suitable;
- reuse DRUNet wrapper logic from `zeroshot_l1_pnp.py`;
- reuse noise sweep structure from `eval_noise_sweep.py`.

If changing existing official files, keep changes minimal and explain why.

### 5.3 First Smoke Test

Before full training, the new method must pass a tiny smoke test:

- use a small subset, e.g. 16 or 32 test images;
- use a small maximum fixed-point iteration count;
- confirm no NaN/Inf;
- confirm output shape is correct;
- confirm nonnegative option works;
- confirm fixed-point residual decreases or at least remains bounded;
- confirm PSNR is above LS/SVD.

The first acceptance criterion is not to beat DEQ-MPI. It is to show the new fixed-point map is numerically valid.

## 6. Experiment Plan

### 6.1 Main 2D Benchmark

Primary benchmark:

- 2D simulated/in-house MPI benchmark built from existing `32 x 32` patch data and in-house system matrix.

Primary test set:

- `datasets/testPatches.h5`, 3,730 samples.

Main methods to compare:

- LS/SVD initialization;
- classical \(l^1\) or nonnegative \(l^1\);
- TV or \(l^1+TV\), if already available;
- PnP-ADMM;
- ZeroShot-PnP;
- ZeroShot-\(l^1\)-PnP;
- supervised DEQ-MPI;
- proposed Sparse-PnP-ADMM-DEQ.

Primary metrics:

- PSNR;
- NRMSE;
- HFEN;
- SSIM, if implemented;
- data residual;
- fixed-point residual;
- runtime;
- GPU memory, if practical.

### 6.2 Stress Tests

Stress tests are required for the paper story.

#### 6.2.1 Low-SNR Robustness

Evaluate pSNR settings:

- `15, 10, 5, 0, -5`

Use existing `eval_noise_sweep.py` as reference.

Question to answer:

> Does the proposed method maintain reconstruction quality under severe measurement noise?

#### 6.2.2 System Matrix Mismatch

Use mismatched forward operators:

- generate data with one system matrix;
- reconstruct with a reduced/interpolated/blurred/mismatched system matrix.

Question to answer:

> Does explicit sparse regularization plus denoising improve robustness when \(A\) is imperfect?

#### 6.2.3 Frequency or Row Undersampling

Simulate reduced measurements by selecting subsets of measurement rows/frequencies.

Suggested levels:

- 100%;
- 75%;
- 50%;
- 25%;
- 10%, if stable.

Question to answer:

> Does the dual-prior DEQ framework improve reconstruction when measurements are incomplete?

### 6.3 Ablation Studies

Required ablations:

- no \(l^1\) branch;
- no denoiser branch;
- no nonnegativity constraint;
- fixed finite unrolling vs DEQ;
- HQS-like update vs ADMM update;
- identity sparsifying transform vs wavelet/TV, if added;
- different \(\lambda\);
- different \(\rho\);
- different \(\eta\);
- different maximum fixed-point iterations;
- Gaussian denoiser vs learned RDN vs DRUNet, if feasible.

The ablation table must show why each component exists.

### 6.4 OpenMPI Real-Data Validation

OpenMPI experiments should be used as realistic transfer evidence, not as the main supervised benchmark.

Available phantoms:

- shape;
- resolution;
- concentration.

Available converted systems:

- `shape_smoke_real_system.npz`;
- `shape_medium_real_system.npz`;
- `shape_large_real_system.npz`;
- `resolution_medium_real_system.npz`;
- `concentration_medium_real_system.npz`.

Report:

- center slices;
- MIP views;
- data residual;
- mean intensity;
- max intensity;
- p95/p99 intensity;
- nonzero ratio;
- qualitative artifact suppression.

Do not overstate OpenMPI results as ground-truth quantitative superiority unless matching ground truth or validated phantom geometry metrics are introduced.

## 7. Figure and Table Plan

### Figure 1: Method Overview

Show:

- MPI forward model \(y=Ax+n\);
- ADMM variables \(x,z,s,u,v\);
- denoiser branch;
- \(l^1\) branch;
- DEQ fixed-point solver.

### Figure 2: Fixed-Point Map

Show:

- one iteration of \(T_\theta\);
- DEQ solving \(w^\ast=T_\theta(w^\ast)\);
- implicit gradient concept.

### Figure 3: Main 2D Reconstruction

Compare:

- ground truth;
- LS/SVD;
- DEQ-MPI;
- ZeroShot-PnP;
- ZeroShot-\(l^1\)-PnP;
- proposed Sparse-PnP-ADMM-DEQ.

### Figure 4: Robustness Curves

Show:

- PSNR vs input pSNR;
- NRMSE vs input pSNR;
- optional residual vs pSNR.

### Figure 5: Ablation Study

Show:

- component ablations;
- parameter sensitivity;
- fixed-point residual behavior.

### Figure 6: OpenMPI Real Phantom Results

Show:

- shape/resolution/concentration phantoms;
- axial slices and MIPs;
- selected method comparison.

### Table 1: Main 2D Quantitative Results

Columns:

- method;
- PSNR mean/std;
- NRMSE;
- HFEN;
- SSIM if available;
- runtime.

### Table 2: Robustness and Ablation

Columns:

- variant;
- low-SNR result;
- mismatch result;
- fixed-point residual;
- parameter setting.

### Table 3: OpenMPI Real-Data Summary

Columns:

- phantom;
- method;
- residual;
- mean;
- max;
- p95;
- p99;
- nonzero ratio.

## 8. Milestones

### Milestone 1: Baseline Freeze

Goal:

- preserve current reproduction results as baseline.

Deliverables:

- baseline summary CSV;
- current DEQ-MPI checkpoint path;
- current 2D summary;
- current OpenMPI summary.

Acceptance:

- DEQ-MPI reproduces approximately `32.2 dB` on the 2D test set.

### Milestone 2: Proposed Fixed-Point Module

Goal:

- implement `SparsePnPADMMDEQ`.

Deliverables:

- `sparse_pnp_admm_deq.py`;
- smoke-test script or smoke-test mode;
- residual logging.

Acceptance:

- no NaN/Inf;
- output shape correct;
- PSNR above LS/SVD on a small subset;
- residual is finite and inspectable.

### Milestone 3: Training Integration

Goal:

- train or fine-tune the proposed model on the existing 2D benchmark.

Deliverables:

- training entry point;
- checkpoint directory;
- validation log.

Acceptance:

- model trains end-to-end;
- validation PSNR is competitive with existing branches;
- training does not overwrite official checkpoints.

### Milestone 4: Full 2D Evaluation

Goal:

- compare all main methods on the full test set.

Deliverables:

- JSON summary;
- CSV table;
- representative figure.

Acceptance:

- complete evaluation on 3,730 test samples;
- comparison includes DEQ-MPI and ZeroShot branches.

### Milestone 5: Stress Tests

Goal:

- test robustness under low SNR, mismatch, and undersampling.

Deliverables:

- robustness curves;
- CSV/JSON results;
- selected visual examples.

Acceptance:

- at least one stress-test setting shows a clear benefit or interpretable tradeoff of the proposed method.

### Milestone 6: OpenMPI Validation

Goal:

- produce real-data transfer evidence.

Deliverables:

- OpenMPI reconstructions;
- MIP/slice figures;
- residual/intensity summary table.

Acceptance:

- results are visually interpretable;
- claims remain qualitative or phantom-geometry based unless ground truth is available.

### Milestone 7: Manuscript Draft

Goal:

- produce SCI manuscript draft.

Deliverables:

- abstract;
- introduction;
- methods;
- experiments;
- results;
- discussion;
- limitations.

Acceptance:

- every main claim maps to an experiment;
- no unsupported novelty claims;
- limitations state OpenMPI and sparsity-prior boundaries clearly.

## 9. Decision Rules

### 9.1 If \(l^1\) Reduces PSNR

Do not hide it.

Interpretation options:

- \(l^1\) suppresses weak vessel-like structures;
- sparse priors are target-distribution dependent;
- \(l^1\) may help sparse phantoms or real OpenMPI transfer more than dense simulated vessels.

Then position the method as controllable and robust rather than universally PSNR-superior.

### 9.2 If Proposed DEQ Does Not Beat DEQ-MPI

Do not force the claim.

Possible revised claims:

- improved low-SNR robustness;
- improved mismatch robustness;
- better controllability;
- better sparsity/false-positive suppression;
- useful hybrid supervised/zero-shot framework;
- explicit prior interpretability.

### 9.3 If Training Is Unstable

First check:

- step sizes \(\rho,\eta,\lambda\);
- nonnegative projection;
- denoiser scaling;
- fixed-point residual;
- Anderson tolerance;
- state normalization;
- whether \(A^TA+\rho I+\eta I\) is ill-conditioned.

Then reduce complexity:

- start with Gaussian denoiser;
- start with fixed denoiser weights;
- start without learned consistency;
- start with finite unrolling before DEQ.

### 9.4 If OpenMPI Results Are Weak

Do not make OpenMPI the main benchmark.

Use it as:

- real-data feasibility;
- transfer evidence;
- qualitative phantom reconstruction;
- motivation for future 3D training.

## 10. Engineering Rules

### 10.1 Reproducibility

Every experiment script should write:

- command-line arguments;
- checkpoint path;
- dataset path;
- random seed;
- method parameters;
- summary JSON;
- per-sample CSV where feasible.

### 10.2 File Safety

Do not overwrite:

- official DEQ-MPI checkpoints;
- raw `.mdf` data;
- original `.mat` files;
- existing reproduction summaries unless explicitly refreshing them.

New artifacts should go under:

- `training/reproduction/sparse_pnp_admm_deq/`

### 10.3 Metrics

For 2D supervised/simulated data:

- PSNR;
- NRMSE;
- HFEN;
- SSIM if implemented;
- residual;
- runtime.

For real OpenMPI data:

- residual;
- mean intensity;
- max intensity;
- p95;
- p99;
- nonzero ratio;
- qualitative MIP/slice assessment.

### 10.4 Documentation

Any new major script should have:

- short top-level purpose comment;
- command example in a Markdown note or docstring;
- output file description.

## 11. Manuscript Claim-Evidence Map

### Claim 1

Sparse-PnP-ADMM-DEQ provides a unified fixed-point reconstruction framework for MPI.

Evidence required:

- method derivation;
- fixed-point state definition;
- implementation diagram;
- residual convergence curves.

### Claim 2

Explicit sparsity and implicit denoising provide complementary control over MPI reconstructions.

Evidence required:

- no-\(l^1\) ablation;
- no-denoiser ablation;
- parameter sensitivity;
- visual comparison.

### Claim 3

DEQ solving is preferable to naive long unrolling under selected conditions.

Evidence required:

- unrolled vs DEQ comparison;
- memory or runtime comparison;
- fixed-point residual;
- performance table.

### Claim 4

The framework is robust under low-SNR, mismatch, or undersampling.

Evidence required:

- stress-test curves;
- quantitative comparison;
- selected reconstruction examples.

### Claim 5

The method has realistic transfer potential for OpenMPI phantom data.

Evidence required:

- OpenMPI MIP/slice figures;
- residual and intensity summaries;
- careful limitation statement.

## 12. Immediate Next Actions

The next development steps should be:

1. Freeze current baseline results and record exact checkpoint paths.
2. Create `sparse_pnp_admm_deq.py`.
3. Implement state packing/unpacking for \(w=(x,z,s,u,v)\).
4. Implement the \(x\)-update, denoiser update, \(l^1\) update, and dual updates.
5. Add fixed-point residual logging.
6. Run a small smoke test on 16 to 32 2D test samples.
7. Compare against LS/SVD and DEQ-MPI on the same subset.
8. Only after stability, integrate training and full test-set evaluation.

## 13. Working Contract

Future work should follow this order:

1. preserve existing reproduction results;
2. implement the true integrated Sparse-PnP-ADMM-DEQ method;
3. validate numerically with smoke tests;
4. run full 2D quantitative evaluation;
5. run stress tests;
6. run OpenMPI real-data validation;
7. write manuscript sections only after the supporting experiments exist.

If a future user instruction conflicts with this document, follow the user instruction for that turn. If the change affects the research direction, update this document before continuing major implementation or manuscript work.

## 14. Progress Update: 2026-07-05

Completed implementation and validation milestones:

1. Added the integrated Sparse-PnP-ADMM-DEQ fixed-point module in `sparse_pnp_admm_deq.py`.
2. Added training and checkpoint evaluation entry points:
   - `train_sparse_pnp_admm_deq.py`
   - `eval_sparse_pnp_admm_deq.py`
   - `eval_sparse_pnp_admm_deq_checkpoint.py`
3. Scaled from smoke/validation subsets to full train, full validation, and full test evaluation.
4. Current best checkpoint:
   - run: `training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10`
   - checkpoint: `best.pth`
   - standalone full-test mean PSNR: `32.0062 dB`
5. Added paired Sparse-PnP-DEQ vs DEQ-MPI evaluation in `eval_paired_sparse_pnp_deqmpi.py`.
6. Paired canonical evaluation on seeds 2026-2028 shows practical equivalence to DEQ-MPI:
   - mean delta Sparse-PnP minus DEQ-MPI: `+0.0018 dB`
   - range: `-0.0029` to `+0.0064 dB`
   - mean Sparse win rate: `49.91%`
   - mean wall-time speedup over DEQ-MPI: `2.60x`
7. Paired canonical `maxIter` sweep on seed 2026:
   - `maxIter=6`: `-1.4532 dB` vs DEQ-MPI, not acceptable;
   - `maxIter=8`: `-0.4026 dB`, still too much quality loss;
   - `maxIter=10`: `-0.0990 dB`, useful fast-inference setting, `3.33x` speedup;
   - `maxIter=12`: `-0.0029 dB`, quality-equivalent setting, `2.62x` speedup.
8. Paired canonical `maxIter=10` eta sweep on seed 2026:
   - best tested setting: `eta=0.001`;
   - `eta=0.001` delta vs DEQ-MPI: `-0.0688 dB`;
   - gain over `eta=0.10`: `+0.0302 dB`;
   - `eta=0` is worse than `eta=0.001`, so a tiny sparse-branch weight is better than fully removing it.
9. Multi-seed fast-setting validation for `maxIter=10`, `eta=0.001`, canonical seeds 2026-2028:
   - mean delta Sparse-PnP minus DEQ-MPI: `-0.0554 dB`;
   - range: `-0.0688` to `-0.0410 dB`;
   - mean Sparse win rate: `46.06%`;
   - mean wall-time speedup over DEQ-MPI: `3.18x`;
   - mean gap vs the `maxIter=12`, `eta=0.10` quality setting: `-0.0572 dB`.
10. Added a reusable quality/runtime figure script:
   - script: `make_sparse_pnp_deq_runtime_figures.py`;
   - figure base: `training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10/figures/sparse_pnp_deq_quality_runtime`;
   - exported formats: SVG, PDF, PNG, TIFF;
   - source data: `sparse_pnp_deq_quality_runtime_source_data.csv`;
   - notes: `sparse_pnp_deq_quality_runtime_notes.md`.
11. Added full-test convergence trace evidence:
   - script: `eval_sparse_pnp_deq_convergence_trace.py`;
   - output directory: `training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10/convergence_traces`;
   - full-test trace: `convergence_canonical_seed2026_full_solver_trace.csv`;
   - per-sample final residuals: `convergence_canonical_seed2026_full_sample_final_residuals.csv`;
   - quality setting final mean Anderson residual: `0.012652`;
   - fast setting final mean Anderson residual: `0.023828`.
12. Refreshed `sparse_pnp_deq_quality_runtime` so panel D now shows full-test batch-averaged Anderson residual traces with q10-q90 bands. Remaining caveat: panel D residuals are batch-level solver residuals, not per-sample residual trajectories.

Important diagnostic note:

- Separate per-sample CSVs from different scripts should not be treated as strict paired comparisons unless target scaling, noise generation, and PSNR peak handling are generated in the same run. Use `eval_paired_sparse_pnp_deqmpi.py` for fair paired claims.

Next recommended steps:

1. Run stability ablations around the current best checkpoint:
   - `eta` around `0.08-0.15`;
   - finite vs DEQ solver under the same paired evaluator.
2. Preserve `maxIter=12`, `eta=0.10` as the main quality setting and `maxIter=10`, `eta=0.001` as the fast-inference setting.
3. Run minimum ablations for manuscript support: no/tiny l1, frozen denoiser, finite vs DEQ solver.
4. Only after paired stability is frozen, move to low-SNR or mismatch stress tests.
