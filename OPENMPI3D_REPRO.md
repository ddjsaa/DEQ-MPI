# 3D OpenMPI ZeroShot-PnP Reproduction

This stage connects the ZeroShot-PnP implementation to the public 3D OpenMPI MDF files. In the overall manuscript plan, this is the zero-shot generalization branch built around the DEQ-MPI fixed-point reconstruction viewpoint, not a replacement for the supervised DEQ-MPI baseline. The DEQ-MPI reproduction and paper integration plan are summarized in `DEQ_MPI_ZEROSHOT_PAPER_PLAN.md`.

## Data

The official files are large:

| File | Local name | Size |
|---|---|---:|
| `calibrations/3.mdf` | `datasets/OpenMPI/calibration_3.mdf` | about 4.45 GB |
| `measurements/shapePhantom/3.mdf` | `datasets/OpenMPI/shape_3.mdf` | about 616 MB |
| `measurements/resolutionPhantom/3.mdf` | `datasets/OpenMPI/resolution_3.mdf` | about 616 MB |
| `measurements/concentrationPhantom/3.mdf` | `datasets/OpenMPI/concentration_3.mdf` | about 616 MB |

Check local download status:

```powershell
python openmpi3d_tools.py status --root datasets/OpenMPI --insecure
```

Download or resume selected files:

```powershell
python openmpi3d_tools.py download --root datasets/OpenMPI --files calibration,shape --insecure
```

Use `--files calibration,shape,resolution,concentration` for all data. The downloader uses `curl --continue-at -`, so interrupted downloads can be resumed.

## Inspect MDF

After a file is complete:

```powershell
python openmpi3d_tools.py inspect datasets/OpenMPI/shape_3.mdf --outJson training/reproduction/openmpi_shape_mdf_tree.json
```

## Convert To A Real Linear System

The MDF data are complex-valued. The converter stacks real and imaginary rows so the particle concentration remains real-valued:

```text
A_real = [real(A); imag(A)]
y_real = [real(y); imag(y)]
```

Smoke conversion with a row limit:

```powershell
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom shape --maxRows 5000 --out datasets/OpenMPI/shape_smoke_real_system.npz
```

Full minimal conversion:

```powershell
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom shape --out datasets/OpenMPI/shape_minimal_real_system.npz
```

The minimal converter performs:

- reads `/measurement/data`
- removes calibration background frames using `/measurement/isBackgroundFrame`
- removes phantom measurement background frames using `/measurement/isBackgroundFrame`
- Fourier-transforms the phantom measurement
- removes frequencies below 80 kHz
- splits complex rows into real and imaginary rows

When `--maxRows` is used, the converter reads only the needed calibration frequency rows. This keeps real-data smoke tests small enough to run on a workstation.

It does not yet implement whitening or randomized SVD. This is intentional for the first integration pass.

## Reconstruct

Run ZeroShot-PnP on the converted 3D system:

```powershell
python eval_zeroshot_openmpi3d.py --npz datasets/OpenMPI/shape_smoke_real_system.npz --denoiser drunet --dpirRoot external/DPIR --drunetWeights external/models/drunet_gray.pth --nIter 7 --mu0 5e5 --alpha 0
```

Run ZeroShot-l1-PnP:

```powershell
python eval_zeroshot_openmpi3d.py --npz datasets/OpenMPI/shape_smoke_real_system.npz --denoiser drunet --dpirRoot external/DPIR --drunetWeights external/models/drunet_gray.pth --nIter 5 --mu0 3e5 --alpha 1500
```

Outputs:

- reconstructed 3D volume: `*_recon.npz`
- metrics/history JSON: `*_recon_metrics.json`

Raw OpenMPI values are large, so `eval_zeroshot_openmpi3d.py` applies RMS scaling to `A` and `y` by default and records `system_scale` in the JSON. Use `--noNormalizeSystem` only for debugging raw-scale behavior.

## Current State

All four public OpenMPI MDF files have been downloaded successfully:

```powershell
python openmpi3d_tools.py status --root datasets/OpenMPI --insecure
```

The downloaded files are:

- `datasets/OpenMPI/calibration_3.mdf`, 4451.42 MB
- `datasets/OpenMPI/shape_3.mdf`, 616.37 MB
- `datasets/OpenMPI/resolution_3.mdf`, 616.37 MB
- `datasets/OpenMPI/concentration_3.mdf`, 616.37 MB

The real `shape` phantom has been inspected and converted with a small row limit:

```powershell
python openmpi3d_tools.py inspect datasets/OpenMPI/calibration_3.mdf --outJson training/reproduction/openmpi_calibration_mdf_tree.json
python openmpi3d_tools.py inspect datasets/OpenMPI/shape_3.mdf --outJson training/reproduction/openmpi_shape_mdf_tree.json
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom shape --maxRows 5000 --out datasets/OpenMPI/shape_smoke_real_system.npz
```

The converted smoke system has:

```text
A: (10000, 6859), float32
y: (10000,), float32
image_shape: (19, 19, 19)
```

The real-data ZeroShot smoke run now completes without NaNs:

```powershell
python eval_zeroshot_openmpi3d.py --npz datasets/OpenMPI/shape_smoke_real_system.npz --gpu -1 --nIter 3 --mu0 500000 --alpha 0 --cg --denoiser gaussian --outRecon training/reproduction/openmpi_shape_smoke_zeroshot_recon.npz --outJson training/reproduction/openmpi_shape_smoke_zeroshot_metrics.json
```

Smoke output summary:

- `system_scale`: 6143.869140625
- elapsed time on CPU: about 0.81 s
- reconstruction range: `[0.0, 0.2272]`
- reconstruction mean: `0.00618`

## Medium Shape Experiment

The first real-data medium-scale experiment uses `shape` with `--maxRows 20000`, which becomes a real-valued system with 40000 rows:

```powershell
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom shape --maxRows 20000 --out datasets/OpenMPI/shape_medium_real_system.npz
```

Converted system:

```text
A: (40000, 6859), float32
y: (40000,), float32
image_shape: (19, 19, 19)
```

Gaussian ZeroShot-PnP and ZeroShot-l1-PnP were run on GPU with `nIter=5`, `mu0=500000`, CG data step, and default OpenMPI RMS normalization.

| Method | alpha | mean | max | nonzero > 1e-4 | p95 | p99 | final residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZeroShot-PnP | 0 | 0.010572 | 0.460140 | 0.122758 | 0.062225 | 0.278956 | 14328.429 |
| ZeroShot-l1-PnP | 1500 | 0.010342 | 0.460909 | 0.113282 | 0.060206 | 0.277522 | 14018.893 |
| ZeroShot-l1-PnP | 5000 | 0.009918 | 0.461876 | 0.110366 | 0.054045 | 0.272072 | 13439.148 |
| ZeroShot-l1-PnP | 15000 | 0.009138 | 0.461847 | 0.132235 | 0.045107 | 0.255868 | 12366.350 |

Artifacts:

- `datasets/OpenMPI/shape_medium_real_system.npz`
- `training/reproduction/openmpi_shape_medium_zeroshot_pnp_gaussian_recon.npz`
- `training/reproduction/openmpi_shape_medium_zeroshot_l1_pnp_gaussian_alpha1500_recon.npz`
- `training/reproduction/openmpi_shape_medium_zeroshot_l1_pnp_gaussian_alpha5000_recon.npz`
- `training/reproduction/openmpi_shape_medium_zeroshot_l1_pnp_gaussian_alpha15000_recon.npz`
- `training/reproduction/openmpi_shape_medium_gaussian_comparison.png`

For the next pass, `alpha=5000` is a useful default. It gives a clearer sparsity/low-intensity suppression effect than `alpha=1500` without the less intuitive nonzero increase seen at `alpha=15000`.

## Medium All-Phantom Gaussian Experiment

The same `--maxRows 20000` conversion was run for `resolution` and `concentration`, giving the same system size as `shape`:

```powershell
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom resolution --maxRows 20000 --out datasets/OpenMPI/resolution_medium_real_system.npz
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom concentration --maxRows 20000 --out datasets/OpenMPI/concentration_medium_real_system.npz
```

For each phantom, Gaussian ZeroShot-PnP (`alpha=0`) was compared with Gaussian ZeroShot-l1-PnP (`alpha=5000`) using `nIter=5`, `mu0=500000`, CG, GPU, and default RMS system normalization.

| Phantom | Method | alpha | mean | max | nonzero > 1e-4 | p95 | p99 | final residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shape | PnP | 0 | 0.010572 | 0.460140 | 0.122758 | 0.062225 | 0.278956 | 14328.429 |
| shape | l1-PnP | 5000 | 0.009918 | 0.461876 | 0.110366 | 0.054045 | 0.272072 | 13439.148 |
| resolution | PnP | 0 | 0.000928 | 0.043730 | 0.171454 | 0.004931 | 0.023975 | 1094.843 |
| resolution | l1-PnP | 5000 | 0.000737 | 0.037620 | 0.230063 | 0.003510 | 0.017422 | 833.125 |
| concentration | PnP | 0 | 0.000505 | 0.019421 | 0.241435 | 0.002641 | 0.009058 | 496.798 |
| concentration | l1-PnP | 5000 | 0.000367 | 0.011137 | 0.287505 | 0.001816 | 0.006093 | 325.370 |

Artifacts:

- `datasets/OpenMPI/resolution_medium_real_system.npz`
- `datasets/OpenMPI/concentration_medium_real_system.npz`
- `training/reproduction/openmpi_medium_gaussian_summary.csv`
- `training/reproduction/openmpi_medium_gaussian_all_phantoms.png`

Across all three phantoms, `alpha=5000` lowers mean intensity, high-percentile intensity, and final data residual. The simple `nonzero > 1e-4` count is less reliable for low-signal phantoms because lower peak intensity can move many small voxels around the fixed threshold. Visual inspection and percentile statistics are better criteria for this stage.

## Medium Gaussian vs DRUNet Experiment

The DPIR DRUNet gray denoiser was then run on the same three medium systems. This creates a 2x2 comparison for each phantom:

```text
denoiser: Gaussian, DRUNet
regularizer: PnP alpha=0, l1-PnP alpha=5000
```

All runs used `nIter=5`, `mu0=500000`, CG, GPU, and default RMS system normalization.

| Phantom | Denoiser | Method | alpha | mean | max | nonzero > 1e-4 | p95 | p99 | final residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| shape | gaussian | PnP | 0 | 0.010572 | 0.460140 | 0.122758 | 0.062225 | 0.278956 | 14328.429 |
| shape | gaussian | l1-PnP | 5000 | 0.009918 | 0.461876 | 0.110366 | 0.054045 | 0.272072 | 13439.148 |
| shape | drunet | PnP | 0 | 0.010934 | 0.510794 | 0.072605 | 0.039445 | 0.332894 | 14862.960 |
| shape | drunet | l1-PnP | 5000 | 0.010289 | 0.516903 | 0.068377 | 0.026614 | 0.328092 | 13983.312 |
| resolution | gaussian | PnP | 0 | 0.000928 | 0.043730 | 0.171454 | 0.004931 | 0.023975 | 1094.843 |
| resolution | gaussian | l1-PnP | 5000 | 0.000737 | 0.037620 | 0.230063 | 0.003510 | 0.017422 | 833.125 |
| resolution | drunet | PnP | 0 | 0.000964 | 0.050920 | 0.157749 | 0.003442 | 0.029605 | 1139.655 |
| resolution | drunet | l1-PnP | 5000 | 0.000802 | 0.047737 | 0.348739 | 0.002454 | 0.020972 | 914.126 |
| concentration | gaussian | PnP | 0 | 0.000505 | 0.019421 | 0.241435 | 0.002641 | 0.009058 | 496.798 |
| concentration | gaussian | l1-PnP | 5000 | 0.000367 | 0.011137 | 0.287505 | 0.001816 | 0.006093 | 325.370 |
| concentration | drunet | PnP | 0 | 0.000550 | 0.024516 | 0.246100 | 0.002326 | 0.011358 | 544.622 |
| concentration | drunet | l1-PnP | 5000 | 0.000429 | 0.013291 | 0.373815 | 0.001569 | 0.007331 | 393.660 |

Artifacts:

- `training/reproduction/openmpi_medium_gaussian_vs_drunet_summary.csv`
- `training/reproduction/openmpi_medium_gaussian_vs_drunet_all_phantoms.png`
- `training/reproduction/openmpi_shape_medium_zeroshot_pnp_drunet_recon.npz`
- `training/reproduction/openmpi_shape_medium_zeroshot_l1_pnp_drunet_alpha5000_recon.npz`
- `training/reproduction/openmpi_resolution_medium_zeroshot_pnp_drunet_recon.npz`
- `training/reproduction/openmpi_resolution_medium_zeroshot_l1_pnp_drunet_alpha5000_recon.npz`
- `training/reproduction/openmpi_concentration_medium_zeroshot_pnp_drunet_recon.npz`
- `training/reproduction/openmpi_concentration_medium_zeroshot_l1_pnp_drunet_alpha5000_recon.npz`

DRUNet changes the morphology more strongly than Gaussian: it lowers p95 but often raises p99/max, concentrating intensity into fewer stronger regions. Adding the l1 branch lowers final residual and p95 for both denoisers. For a paper-grade result, this stage should be followed by visual scoring and phantom-specific alpha tuning instead of relying on a fixed threshold count.

## Multi-View Qualitative Exports

`export_openmpi_views.py` exports axial, coronal, sagittal, and maximum-intensity-projection views from a reconstructed 3D volume. It can export one reconstruction or a full 2x2 grid for a phantom:

```powershell
python export_openmpi_views.py --phantom shape --outDir training/reproduction/openmpi_views
python export_openmpi_views.py --phantom resolution --outDir training/reproduction/openmpi_views
python export_openmpi_views.py --phantom concentration --outDir training/reproduction/openmpi_views
```

Artifacts:

- `training/reproduction/openmpi_views/openmpi_shape_medium_views_grid.png`
- `training/reproduction/openmpi_views/openmpi_resolution_medium_views_grid.png`
- `training/reproduction/openmpi_views/openmpi_concentration_medium_views_grid.png`

These figures are more suitable for paper drafting than the earlier center-slice-only overview because each method is shown with three orthogonal slices plus MIP.

## DRUNet Alpha Sweep

DRUNet alpha sweeps were run for the two lower-signal phantoms:

```powershell
python sweep_openmpi3d_alpha.py --phantoms resolution,concentration --alphas 1000,2500,5000,10000 --denoiser drunet --gpu 0 --nIter 5 --mu0 500000 --summaryCsv training/reproduction/openmpi_medium_drunet_alpha_sweep_resolution_concentration.csv
```

| Phantom | alpha | mean | max | p95 | p99 | nonzero > 1e-4 | final residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| resolution | 1000 | 0.000853 | 0.051890 | 0.001733 | 0.028301 | 0.175098 | 989.701 |
| resolution | 2500 | 0.000809 | 0.052395 | 0.001817 | 0.025025 | 0.274238 | 925.972 |
| resolution | 5000 | 0.000802 | 0.047737 | 0.002454 | 0.020972 | 0.348739 | 914.126 |
| resolution | 10000 | 0.000796 | 0.032726 | 0.002867 | 0.019620 | 0.376002 | 905.201 |
| concentration | 1000 | 0.000457 | 0.024095 | 0.001432 | 0.009568 | 0.308937 | 423.990 |
| concentration | 2500 | 0.000436 | 0.020258 | 0.001573 | 0.007509 | 0.355883 | 400.807 |
| concentration | 5000 | 0.000429 | 0.013291 | 0.001569 | 0.007331 | 0.373815 | 393.660 |
| concentration | 10000 | 0.000429 | 0.013289 | 0.001569 | 0.007331 | 0.373815 | 393.654 |

Artifacts:

- `training/reproduction/openmpi_medium_drunet_alpha_sweep_resolution_concentration.csv`
- `training/reproduction/openmpi_medium_drunet_alpha_sweep_trends.png`

Interpretation:

- `resolution`: increasing alpha continues to lower final residual and p99, but p95 and fixed-threshold nonzero ratio increase. A reasonable next visual comparison is `alpha=2500` vs `alpha=10000`.
- `concentration`: the curve nearly saturates at `alpha=5000`; `alpha=10000` gives no meaningful improvement.
- A fixed threshold like `1e-4` is not a stable sparsity proxy across phantoms. Percentiles and qualitative MIP/slice views should drive the next selection.

## Shape Large-Row Validation

The `shape` phantom was then expanded from `--maxRows 20000` to `--maxRows 50000`:

```powershell
python openmpi3d_tools.py convert --root datasets/OpenMPI --phantom shape --maxRows 50000 --out datasets/OpenMPI/shape_large_real_system.npz
```

Converted system:

```text
A: (100000, 6859), float32
y: (100000,), float32
image_shape: (19, 19, 19)
```

The same 2x2 comparison was run with Gaussian and DRUNet, `alpha=0` and `alpha=5000`.

| Scale | Rows | Denoiser | Method | alpha | mean | max | nonzero > 1e-4 | p95 | p99 | final residual |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 40000 | gaussian | PnP | 0 | 0.010572 | 0.460140 | 0.122758 | 0.062225 | 0.278956 | 14328.429 |
| medium | 40000 | gaussian | l1-PnP | 5000 | 0.009918 | 0.461876 | 0.110366 | 0.054045 | 0.272072 | 13439.148 |
| medium | 40000 | drunet | PnP | 0 | 0.010934 | 0.510794 | 0.072605 | 0.039445 | 0.332894 | 14862.960 |
| medium | 40000 | drunet | l1-PnP | 5000 | 0.010289 | 0.516903 | 0.068377 | 0.026614 | 0.328092 | 13983.312 |
| large | 100000 | gaussian | PnP | 0 | 0.011184 | 0.490909 | 0.117656 | 0.066700 | 0.288671 | 23961.451 |
| large | 100000 | gaussian | l1-PnP | 5000 | 0.010718 | 0.492937 | 0.105263 | 0.060875 | 0.285443 | 22958.980 |
| large | 100000 | drunet | PnP | 0 | 0.011567 | 0.541606 | 0.073626 | 0.041984 | 0.356473 | 24855.121 |
| large | 100000 | drunet | l1-PnP | 5000 | 0.011094 | 0.546821 | 0.070127 | 0.031736 | 0.354988 | 23835.627 |

Artifacts:

- `datasets/OpenMPI/shape_large_real_system.npz`
- `training/reproduction/openmpi_shape_medium_vs_large_summary.csv`
- `training/reproduction/openmpi_views/openmpi_shape_large_views_grid.png`
- `training/reproduction/openmpi_shape_medium_vs_large_center_slices.png`
- `training/reproduction/openmpi_shape_large_zeroshot_pnp_gaussian_recon.npz`
- `training/reproduction/openmpi_shape_large_zeroshot_l1_pnp_gaussian_alpha5000_recon.npz`
- `training/reproduction/openmpi_shape_large_zeroshot_pnp_drunet_recon.npz`
- `training/reproduction/openmpi_shape_large_zeroshot_l1_pnp_drunet_alpha5000_recon.npz`

The medium-scale conclusions remain stable after increasing row count: the l1 branch lowers mean intensity, p95, fixed-threshold nonzero ratio, and final residual. DRUNet still produces lower p95 but higher p99/max, suggesting a more concentrated high-intensity reconstruction.

## Shape Large DRUNet Alpha Sweep

The large `shape` system was then swept with DRUNet and `alpha = 1000, 2500, 5000, 10000`:

```powershell
python sweep_openmpi3d_alpha.py --phantoms shape --scale large --alphas 1000,2500,5000,10000 --denoiser drunet --gpu 0 --nIter 5 --mu0 500000 --summaryCsv training/reproduction/openmpi_shape_large_drunet_alpha_sweep.csv
```

| alpha | mean | max | p95 | p99 | nonzero > 1e-4 | final residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.011475 | 0.542512 | 0.039741 | 0.356361 | 0.073189 | 24657.719 |
| 2500 | 0.011329 | 0.543963 | 0.036594 | 0.355962 | 0.072022 | 24345.803 |
| 5000 | 0.011094 | 0.546821 | 0.031736 | 0.354988 | 0.070127 | 23835.627 |
| 10000 | 0.010668 | 0.553362 | 0.023972 | 0.352624 | 0.069689 | 22909.568 |

Artifacts:

- `training/reproduction/openmpi_shape_large_drunet_alpha_sweep.csv`
- `training/reproduction/openmpi_shape_large_drunet_alpha_sweep_trends.png`
- `training/reproduction/openmpi_shape_large_drunet_alpha_sweep_views.png`
- `training/reproduction/openmpi_shape_large_zeroshot_l1_pnp_drunet_alpha1000_recon.npz`
- `training/reproduction/openmpi_shape_large_zeroshot_l1_pnp_drunet_alpha2500_recon.npz`
- `training/reproduction/openmpi_shape_large_zeroshot_l1_pnp_drunet_alpha5000_recon.npz`
- `training/reproduction/openmpi_shape_large_zeroshot_l1_pnp_drunet_alpha10000_recon.npz`

On the large `shape` system, higher alpha monotonically lowers mean intensity, p95, p99, fixed-threshold nonzero ratio, and final residual. The tradeoff is that max intensity increases, so `alpha=10000` gives the strongest concentration. `alpha=5000` remains a conservative default, while `alpha=10000` is the stronger candidate if visual inspection confirms that the high-intensity concentration matches the phantom geometry.

## Paper Figure Exports

`make_openmpi_paper_figures.py` exports the current manuscript-ready figure candidates:

```powershell
python make_openmpi_paper_figures.py
```

Selected configurations:

- `shape`: large system, Gaussian PnP, Gaussian l1-PnP `alpha=5000`, DRUNet PnP, DRUNet l1-PnP `alpha=10000`
- `resolution`: medium system, Gaussian PnP, Gaussian l1-PnP `alpha=5000`, DRUNet PnP, DRUNet l1-PnP `alpha=10000`
- `concentration`: medium system, Gaussian PnP, Gaussian l1-PnP `alpha=5000`, DRUNet PnP, DRUNet l1-PnP `alpha=5000`

Artifacts:

- `training/reproduction/paper_figures/openmpi_main_selected_methods.png`
- `training/reproduction/paper_figures/openmpi_drunet_alpha_selection_mip.png`
- `training/reproduction/paper_figures/openmpi_selected_paper_results.csv`

The main figure is exported at `3600 x 2700` pixels. It shows axial center slices and MIPs for each selected configuration, with row-wise intensity scaling per phantom. The alpha selection figure is exported at `3300 x 2100` pixels and compares DRUNet-l1 MIPs across `alpha = 1000, 2500, 5000, 10000`.

Next recommended experiments:

- decide whether `resolution` and `concentration` need large-row validation or whether medium rows are sufficient for the manuscript comparison
- choose between `alpha=5000` and `alpha=10000` for the main `shape` figure after visual inspection
- start drafting the methods/results text around the selected figure and table
