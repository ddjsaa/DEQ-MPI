# Project Freeze Status

Date: 2026-07-05

This file freezes the current local state of the DEQ-MPI reproduction and
Sparse-PnP-ADMM-DEQ extension work. It is intended to guide the next cleanup,
commit, and manuscript-preparation steps without rerunning experiments blindly.

## Executive Status

The project is past the exploratory stage. The official DEQ-MPI reproduction is
complete, the Sparse-PnP-ADMM-DEQ method has a working implementation, and the
main evidence package already includes paired 2D evaluation, ablation, stress
tests, OpenMPI 3D feasibility validation, and a manuscript-style draft.

Approximate status:

- Reproduction engineering: complete.
- Sparse-PnP-ADMM-DEQ implementation: complete enough for paper experiments.
- Evidence package: mostly complete, with claim boundaries clearly identified.
- Submission readiness: not complete; cleanup, citation work, and manuscript
  polishing remain.

## Completed Work

### Official DEQ-MPI reproduction

The reproduced DEQ-MPI checkpoint matches the paper-scale result.

- Final reproduced test mean pSNR: 32.193039 dB.
- Test pSNR standard deviation: 4.514024 dB.
- Paper Table I reference: 32.1 +/- 4.5 dB.
- Main record: `REPRO_RESULTS.md`.
- Full narrative: `FINAL_REPRODUCTION_REPORT.md`.
- Environment notes: `REPRODUCTION.md`.

### Engineering fixes for reproducibility

The training and inference path was adapted for local modern CUDA/PyTorch use
without overwriting the authors' supplied checkpoints.

Important changes include:

- Reproduction outputs routed under `training/reproduction/`.
- Short-run scheduler edge cases fixed.
- Modern PyTorch in-place autograd issues fixed.
- DEQ-MPI resume arguments added.
- Periodic checkpoint naming made more consistent with actual epoch progress.
- Single-checkpoint evaluation entry point added.

### Sparse-PnP-ADMM-DEQ implementation

The proposed method is implemented in `sparse_pnp_admm_deq.py`.

Implemented components:

- Fixed-point state packing for `w = (x, z, s, u, v)`.
- Linear data-consistency update.
- Plug-and-play denoiser branch.
- Non-negative l1 proximal branch.
- Dual variable updates.
- Fixed-point diagnostics.
- Finite-iteration inference.
- DEQ wrapper with fixed-point and Anderson solver support.

Primary training/evaluation entry points:

- `train_sparse_pnp_admm_deq.py`
- `eval_sparse_pnp_admm_deq.py`
- `eval_sparse_pnp_admm_deq_checkpoint.py`
- `eval_paired_sparse_pnp_deqmpi.py`
- `eval_sparse_pnp_deq_convergence_trace.py`

### Main evidence package

The current strongest claim is bounded and reproducible:

> Sparse-PnP-ADMM-DEQ reaches DEQ-MPI-level quality on the paired canonical 2D
> benchmark while offering faster inference settings.

Headline numbers from the quality/runtime evidence package:

- Quality setting canonical mean delta vs DEQ-MPI: +0.0018 dB.
- Fast setting canonical mean delta vs DEQ-MPI: -0.0554 dB.
- Fast setting mean speedup: about 3.18x.
- Quality final mean solver residual: 0.012652.
- Fast final mean solver residual: 0.023828.

Important figure artifacts are under:

`training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10/figures/`

The key manuscript-facing draft is:

- `SPARSE_PNP_DEQ_MANUSCRIPT_DRAFT.md`

## Claim Boundaries

The following claims are supported:

- The official DEQ-MPI result has been locally reproduced.
- Sparse-PnP-DEQ matches DEQ-MPI quality under the main paired canonical setting.
- Sparse-PnP-DEQ has faster inference operating points.
- The denoiser branch is essential.
- Supervised Sparse-PnP tuning matters.
- SVD-rank robustness is strong across the tested ranks 180, 220, and 250.
- OpenMPI 3D transfer is feasible as qualitative/diagnostic real-data evidence.

The following claims should not be made without additional evidence:

- Do not claim broad PSNR superiority over DEQ-MPI.
- Do not claim that the l1 branch is a major mean-PSNR driver in the current
  benchmark.
- Do not treat OpenMPI 3D as a supervised quantitative benchmark.
- Do not claim broad noise robustness across all pSNR settings; pSNR 8 and 15
  currently show visible gaps.
- Do not claim DEQ universally improves PSNR over finite unrolling.

## Git Hygiene Status

The repository currently contains many useful uncommitted changes. They should
be staged intentionally rather than all at once.

Current cleanup decisions already applied:

- `.gitignore` now excludes local external dependencies, model weights,
  checkpoints, reproduction outputs, logs, and virtual environments.
- `.gitattributes` now marks source files as LF-normalized text and common
  artifact formats as binary.

Files that should usually be committed:

- Reproduction documentation:
  - `REPRODUCTION.md`
  - `REPRO_RESULTS.md`
  - `FINAL_REPRODUCTION_REPORT.md`
  - `OPENMPI3D_REPRO.md`
  - `ADMM_DEQ_MPI_PROJECT_PLAN.md`
  - `SPARSE_PNP_DEQ_MANUSCRIPT_DRAFT.md`
  - `PROJECT_FREEZE_STATUS.md`
- Reproduction and evaluation scripts:
  - `eval_repro_deqmpi.py`
  - `summarize_repro_results.py`
  - `make_2d_paper_comparison.py`
  - `eval_noise_sweep.py`
  - `eval_deq_initialized_zeroshot.py`
  - `eval_few_training_proxy.py`
  - `zeroshot_l1_pnp.py`
  - `sparse_pnp_admm_deq.py`
  - `train_sparse_pnp_admm_deq.py`
  - `eval_sparse_pnp_admm_deq.py`
  - `eval_sparse_pnp_admm_deq_checkpoint.py`
  - `eval_paired_sparse_pnp_deqmpi.py`
  - `eval_sparse_pnp_deq_convergence_trace.py`
  - `openmpi3d_tools.py`
  - `eval_zeroshot_openmpi3d.py`
  - `sweep_openmpi3d_alpha.py`
  - `make_openmpi_validation_summary.py`
  - `make_openmpi_paper_figures.py`
  - `make_sparse_pnp_deq_ablation_summary.py`
  - `make_sparse_pnp_deq_runtime_figures.py`
  - `make_sparse_pnp_deq_stress_summary.py`
- Environment file:
  - `requirements-repro-cu128.txt`
- Source-code fixes to existing training/inference files, after review.

Files and directories that should stay local or ignored:

- `.venv/`
- `datasets/`
- `training/reproduction/`
- `logs/`
- `wandb/`
- `external/DPIR/`
- `external/models/`
- `*.pth`, `*.pt`, `*.ckpt`
- regenerated full-size figure exports and checkpoint artifacts.

Tracked generated files that need an explicit decision:

- `Experimental15dB.mat`
- `experimentalOutput15dB.png`

These are already tracked by the upstream repository and were regenerated
locally. Decide whether to commit the refreshed outputs or restore them before
publication cleanup.

Diff review notes for modified upstream files:

- Functional changes are concentrated in output isolation, resume support,
  scheduler safety for short smoke runs, checkpoint save timing, and modern
  PyTorch autograd compatibility.
- Several modified files also contain explanatory Chinese comments. Before a
  clean public commit, decide whether these comments should remain, be converted
  to concise English comments, or be removed where they only narrate obvious
  code.
- Git was warning about line-ending normalization for several Python files.
  `.gitattributes` was added to make future diffs more predictable.

## Recommended Commit Plan

Commit 1: reproducibility infrastructure

- `.gitattributes`
- `.gitignore`
- `requirements-repro-cu128.txt`
- source-code fixes in the official training/inference path
- `REPRODUCTION.md`
- `REPRO_RESULTS.md`
- `FINAL_REPRODUCTION_REPORT.md`
- `eval_repro_deqmpi.py`
- `summarize_repro_results.py`

Commit 2: zero-shot and OpenMPI tooling

- `zeroshot_l1_pnp.py`
- `openmpi3d_tools.py`
- OpenMPI evaluation and figure scripts
- `OPENMPI3D_REPRO.md`

Commit 3: Sparse-PnP-ADMM-DEQ method

- `sparse_pnp_admm_deq.py`
- `train_sparse_pnp_admm_deq.py`
- Sparse-PnP evaluation scripts
- Sparse-PnP summary/figure generation scripts

Commit 4: manuscript and project planning

- `ADMM_DEQ_MPI_PROJECT_PLAN.md`
- `SPARSE_PNP_DEQ_MANUSCRIPT_DRAFT.md`
- `PROJECT_FREEZE_STATUS.md`

## Immediate Next Steps

1. Review diffs in the modified official files and verify they are scoped to
   reproducibility and compatibility fixes.
2. Decide whether regenerated tracked outputs should be committed or restored:
   `Experimental15dB.mat` and `experimentalOutput15dB.png`.
3. Run a quick smoke command after cleanup, preferably one cheap evaluation
   using the existing environment.
4. Stage the first commit group only.
5. Convert `SPARSE_PNP_DEQ_MANUSCRIPT_DRAFT.md` into a target-venue manuscript
   outline with citations, equations, and Methods details.

## Manuscript Priorities

Highest priority:

- Preserve the paired-evaluation framing.
- Lead with quality-speed tradeoff rather than broad superiority.
- State clearly that OpenMPI is feasibility evidence.
- Add citations and exact implementation details.

Additional evidence worth adding only if time allows:

- Support/tail-intensity metrics to make the l1 branch's contribution clearer.
- Noise-aware retuning for pSNR 8 and pSNR 15.
- A cleaner finite-vs-DEQ solver comparison.
- A method schematic and equation-ready Methods section.
