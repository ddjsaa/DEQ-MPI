# Sparse-PnP-ADMM-DEQ Manuscript-Style Consolidation

This draft consolidates the current Sparse-PnP-ADMM-DEQ evidence package into a manuscript-style Results, Methods, Discussion, and Conclusion scaffold. It is intentionally bounded to the experiments that have already been run.

## One-sentence Argument

In the 2D simulated MPI reconstruction benchmark, Sparse-PnP-ADMM-DEQ matches DEQ-MPI-level image quality under paired evaluation while offering faster inference settings, and its denoiser, solver, convergence, stress-test, and OpenMPI 3D transfer analyses define where the method is reliable and where the evidence remains feasibility-level.

## Terminology Ledger

| Canonical term | Definition | Use |
|---|---|---|
| Sparse-PnP-ADMM-DEQ | Proposed sparse plug-and-play ADMM fixed-point model evaluated with DEQ solvers | Use for the method |
| Sparse-PnP-DEQ | Short form for Sparse-PnP-ADMM-DEQ after first use | Use only after first definition |
| DEQ-MPI | Reproduced learned DEQ baseline for MPI reconstruction | Use as the primary baseline |
| quality setting | Sparse-PnP-DEQ with maxIter=12 and eta=0.10 | Use for main quality result |
| fast setting | Sparse-PnP-DEQ with maxIter=10 and eta=0.001 | Use for speed-favoring result |
| paired evaluation | Evaluation where Sparse-PnP-DEQ and DEQ-MPI share the exact target, system, scaling, and noise realization | Use for fair comparison |
| canonical mode | No-inverse-crime evaluation using the canonical reconstructed system | Use for main simulated benchmark |
| direct mode | Direct reconstruction/measurement-system setting | Use for system-mismatch comparison |
| OpenMPI 3D validation | Real-data transfer/feasibility evaluation on OpenMPI shape, resolution, and concentration phantoms | Do not describe as PSNR validation |

## Section Outline

1. Method overview: define the inverse problem, fixed-point state, sparse and denoising branches, and DEQ solver.
2. Experimental protocol: define paired simulated evaluation, metrics, seeds, and baselines.
3. Main 2D result: quality/runtime frontier, multi-seed paired validation, and convergence traces.
4. Ablation: isolate l1 strength, denoiser training, denoiser removal, and finite-vs-DEQ solver behavior.
5. Stress tests: test pSNR, SVD rank, and direct-vs-canonical system changes.
6. OpenMPI 3D validation: present real-data transfer as feasibility evidence.
7. Discussion and conclusion: interpret the tradeoffs and state boundaries.

## Draft

### Method Overview

Sparse-PnP-ADMM-DEQ formulates MPI reconstruction as a fixed-point problem that combines data consistency, plug-and-play denoising, and an explicit sparsity-promoting proximal branch. The method represents each state as a packed ADMM tuple containing the reconstruction variable, denoiser branch, sparse branch, and dual variables. Given reduced MPI measurements and a low-rank system representation, one fixed-point map updates the reconstruction through a linear data-consistency step, applies a learned residual dense-network denoiser to the image branch, applies a non-negative soft-thresholding operator to the sparse branch, and updates the corresponding dual variables. This construction preserves the algorithmic structure of plug-and-play ADMM while making the whole update map compatible with deep equilibrium solvers.

The equilibrium formulation was used for both training and inference. In the main implementation, the fixed-point map was wrapped by the repository DEQ layer with Anderson acceleration. The quality setting used maxIter=12 and eta=0.10, whereas the speed-favoring setting used maxIter=10 and eta=0.001. Both settings used the same trained Sparse-PnP-DEQ checkpoint unless otherwise noted. This separation of method, solver, and inference setting allowed the evaluation to distinguish the contribution of the learned denoiser, explicit sparsity branch, and fixed-point solver.

### Experimental Protocol

The primary 2D validation used a paired evaluation protocol to avoid false gaps caused by independent random scaling or noise generation. For each seed, Sparse-PnP-DEQ and DEQ-MPI were evaluated on the same target patches, system construction, pSNR setting, noise realization, and intensity scaling. The main benchmark used canonical mode, pSNR=10, SVD rank 220, and the full test set of 3730 samples. Reconstruction quality was reported as PSNR and NRMSE, with wall-clock runtime used to estimate speedup against DEQ-MPI. Additional diagnostics recorded fixed-point residuals, data residuals, non-negativity, and per-sample paired deltas.

The evidence package was structured in four layers. First, a full-test paired benchmark quantified quality and runtime under the quality and fast settings. Second, a minimal ablation tested l1 strength, denoiser training, denoiser removal, and finite fixed-point inference. Third, stress tests varied pSNR, SVD rank, and system construction. Fourth, OpenMPI 3D real-data experiments assessed transfer feasibility using visual structure, residual scale, intensity distribution, and support rather than PSNR, because no voxel-level ground truth was available.

### Main 2D Results

Sparse-PnP-DEQ reached DEQ-MPI-level quality under paired canonical evaluation while reducing inference time. Across the paired multi-seed canonical comparison, the quality setting achieved a mean PSNR delta of +0.0018 dB relative to DEQ-MPI, which is effectively an equality result at the reported precision. The same setting provided about 2.60x speedup. The fast setting traded a small quality reduction for additional runtime gain: it was lower than DEQ-MPI by 0.0554 dB on average while reaching about 3.18x speedup.

The eta and iteration sweeps supported the use of two operating points rather than a single universal setting. At maxIter=10, eta=0.001 improved the fast setting by 0.0302 dB compared with eta=0.10, indicating that solver and relaxation parameters affected the quality-speed point even when the trained checkpoint was fixed. This motivates reporting the quality and fast settings separately: the quality setting is the paired-equivalence result, whereas the fast setting is the practical high-throughput option.

The full-test convergence trace gave additional evidence that the DEQ solve was numerically controlled. In canonical full-test evaluation, the quality setting reached a final mean Anderson solver residual of 0.012652, while the fast setting reached 0.023828. These traces were averaged over full-test batches, not per-sample trajectories, so they should be used as solver-level diagnostics rather than as a per-sample convergence proof.

### Ablation Results

The minimal ablation showed that the learned denoiser and supervised Sparse-PnP tuning were the most important contributors among the tested components. Removing the Sparse-PnP checkpoint and using only the pretrained RDN denoiser reduced PSNR by 0.8971 dB relative to the current-l1 quality setting. Replacing the denoiser branch with an identity map caused a much larger drop of 14.8813 dB. These results support the interpretation that Sparse-PnP-DEQ is not merely a data-consistency solver with a weak prior; the denoising branch and its supervised adaptation are essential to the observed reconstruction quality.

In contrast, the l1 strength was weakly influential in the current single-seed ablation. Removing l1 at inference changed PSNR by +0.0017 dB relative to the current-l1 setting, and reducing l1 to 0.0002 changed PSNR by +0.0010 dB. This result does not show that sparsity is useless; rather, it indicates that, for this checkpoint and the canonical pSNR=10 full-test setting, the explicit l1 coefficient was not the dominant determinant of mean PSNR. The l1 branch may still affect support, tail intensity, or robustness in settings not captured by mean PSNR.

The finite-vs-DEQ comparison further clarified the solver tradeoff. Running the same denoiser checkpoint with finite fixed-point iterations at it12 gave +0.0216 dB relative to DEQ-Anderson it12 in the single-seed ablation. This should not be framed as a failure of the DEQ formulation. Instead, it shows that solver choice interacts with the operating point and that the DEQ setting should be justified by its equilibrium formulation, speed behavior, and convergence diagnostics rather than by claiming a universal PSNR advantage over finite unrolling.

### Stress Tests

The stress tests showed stable behavior under SVD-rank changes but a clearer boundary under pSNR shifts. Across SVD ranks 180, 220, and 250, the paired Sparse-PnP-DEQ delta against DEQ-MPI stayed between -0.0087 and +0.0108 dB. This narrow range suggests that the method is not tuned to a single low-rank truncation choice in the tested interval.

The pSNR sweep was less uniform. At the training-noise canonical setting, pSNR=10, Sparse-PnP-DEQ was effectively tied with DEQ-MPI in the single-seed stress table, with a -0.0029 dB delta. At pSNR=12, the gap remained small at -0.0046 dB. However, pSNR=8 and pSNR=15 showed larger gaps of -0.2613 and -0.2185 dB, respectively. These results define a useful boundary: the current checkpoint is strongest near the pSNR regime used for training and validation, while lower- and higher-noise extrapolation may require retraining, parameter retuning, or a noise-aware inference policy.

The direct-vs-canonical comparison showed that system construction also affected the paired gap. Direct mode produced a -0.0546 dB delta, whereas canonical mode produced a -0.0029 dB delta in the single-seed full-test comparison. Because the canonical protocol is the main no-inverse-crime benchmark, the canonical result remains the primary claim; direct mode is best used as a mismatch/control condition rather than the headline result.

### OpenMPI 3D Real-Data Validation

OpenMPI 3D experiments were used as a transfer and feasibility validation rather than a supervised PSNR benchmark. The evaluated real-data phantoms included shape, resolution, and concentration measurements, with selected Gaussian PnP, Gaussian l1-PnP, DRUNet PnP, and DRUNet l1-PnP reconstructions. Since voxel-level ground truth was not available, the evaluation focused on visual structure, final data residual scale, intensity distribution, and relative support.

The OpenMPI reconstructions produced stable localized structures across the selected phantoms and methods. The selected reconstructions were computationally lightweight, with a median runtime of 0.525 s and a maximum runtime of 0.887 s. DRUNet-l1 reduced the p95 intensity relative to DRUNet PnP by 0.571x to 0.833x, depending on the phantom, indicating that the l1-regularized variants suppressed intermediate-intensity background or tail mass. Relative support at a >5% peak threshold shifted by -0.0061 to +0.0668, showing that the effect of l1 regularization was phantom-dependent rather than universally support-shrinking.

The residual trajectories should be interpreted cautiously. The reported ratios compare the final residual with the first recorded residual in the reconstruction history, not with a zero-iteration baseline. These ratios ranged from 1.349 to 2.300 in the selected table, reflecting the data-fit and prior-regularization tradeoff rather than monotonic residual minimization. The alpha sweep showed that alpha=10000 gave the lowest final residual among the tested DRUNet-l1 alpha values for shape, resolution, and concentration phantoms, but alpha selection should still be discussed as a tradeoff among residual, intensity tails, and support.

### Discussion

The current evidence supports a bounded but useful claim: Sparse-PnP-ADMM-DEQ can match DEQ-MPI on the main paired 2D simulated benchmark while offering a faster inference operating point, and its behavior is reproducible across convergence, ablation, stress, and OpenMPI feasibility analyses. The paired evaluation is central to this claim because it removes a major confound from earlier comparisons: independent scripts can produce different target scaling and noise realizations, making small PSNR gaps unreliable.

The ablation results identify the denoiser branch and supervised Sparse-PnP tuning as the strongest contributors to quality. The weak mean-PSNR effect of l1 strength in the current single-seed ablation does not invalidate the sparse branch, but it narrows what can be claimed from the present evidence. The strongest wording is that the sparse branch is part of the method and affects support/intensity tradeoffs, while mean PSNR at the main setting is dominated by the learned denoiser and solver behavior. A stronger sparsity claim would require support-specific metrics, synthetic sparse targets, or real-data structure metrics as primary endpoints.

The stress tests further define the operating boundary. SVD-rank variation was benign in the tested range, but pSNR extrapolation was less stable. This suggests that a noise-aware eta, maxIter, or denoiser policy may be needed before claiming broad noise robustness. The current results are therefore strongest for the canonical pSNR=10 regime and nearby pSNR=12, with pSNR=8 and pSNR=15 treated as visible boundary cases.

The OpenMPI 3D results are encouraging but should remain feasibility evidence. They show that the reconstruction machinery can be applied to real OpenMPI systems and produce visually structured phantoms with measurable intensity and support behavior. They do not establish supervised real-data accuracy, because no voxel-level ground truth is available. A full real-data validation would require physical ground truth, calibrated phantom geometry, or independent measurement-derived endpoints.

### Conclusion

Sparse-PnP-ADMM-DEQ provides a reproducible fixed-point reconstruction framework that reaches DEQ-MPI-level quality on the paired 2D simulated benchmark and offers a faster inference setting with a small quality tradeoff. The strongest evidence is the paired multi-seed canonical result, full-test convergence trace, denoiser ablation, rank stress test, and OpenMPI feasibility package. The main boundaries are also clear: pSNR extrapolation remains less stable, l1-specific benefits require support-oriented endpoints, and real-data validation is currently qualitative and diagnostic rather than ground-truth quantitative.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| Sparse-PnP-DEQ matches DEQ-MPI quality under the main paired canonical setting | Multi-seed quality delta +0.0018 dB | Supported |
| Sparse-PnP-DEQ offers faster inference | Quality speedup about 2.60x; fast speedup about 3.18x | Supported |
| Fast setting trades little quality for speed | Fast setting mean delta -0.0554 dB and 3.18x speedup | Supported |
| Anderson solver reaches controlled full-test residuals | Quality final mean solver residual 0.012652; fast 0.023828 | Supported as batch-level solver diagnostic |
| Denoiser branch is essential | Identity/no-denoiser drop 14.8813 dB | Supported |
| Supervised Sparse-PnP tuning matters | Pretrained RDN without Sparse checkpoint drop 0.8971 dB | Supported |
| l1 coefficient is a major mean-PSNR driver | no-l1 and tiny-l1 changed mean PSNR by about +0.001 to +0.002 dB | Not supported in current single-seed PSNR ablation |
| Rank robustness is strong across tested SVD ranks | Delta range -0.0087 to +0.0108 dB for ranks 180/220/250 | Supported within tested range |
| Noise robustness is broad across pSNR | pSNR=8 and pSNR=15 gaps around -0.22 to -0.26 dB | Needs qualification |
| OpenMPI transfer is feasible | 12 selected real-data reconstructions, visual structure, metrics, median 0.525 s runtime | Supported as feasibility, not PSNR accuracy |

## Figure and Table Mapping

| Manuscript element | Artifact |
|---|---|
| Main 2D quality-runtime figure | `training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10/figures/sparse_pnp_deq_quality_runtime.png` |
| Ablation table | `training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10/ablations/sparse_pnp_deq_minimal_ablation.md` |
| Stress-test figure | `training/reproduction/sparse_pnp_admm_deq/deq_anderson_rdn_full_e3_it12_eta0p10/stress_tests/sparse_pnp_deq_stress_tests.png` |
| OpenMPI validation figure | `training/reproduction/openmpi_validation/openmpi_validation_summary.png` |
| OpenMPI selected-method visual panel | `training/reproduction/paper_figures/openmpi_main_selected_methods.png` |

## Assumptions and Missing Inputs

- Target venue is not specified; this draft uses a generic methods-paper structure.
- The current text does not cite prior work; citations must be inserted before manuscript use.
- Hardware details, exact CUDA/PyTorch versions, and release policy are not yet written into the Methods section.
- OpenMPI validation remains feasibility-level because no voxel-level real-data ground truth is available.
- A stronger sparsity claim needs support-specific endpoints beyond mean PSNR.

## Chinese Notes for Revision

This draft deliberately leads with paired evaluation and speed because those are the strongest supported claims. It avoids saying that the l1 branch improves PSNR, because the current ablation does not support that statement. It also treats OpenMPI as transfer feasibility, not final quantitative validation. To turn this into a submission-ready manuscript, the next revision should add citations, exact implementation details, and a tighter Methods subsection with equations.
