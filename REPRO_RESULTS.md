# DEQ-MPI reproduction results

This file records the final local reproduction results for the DEQ-MPI
repository. All reproduction artifacts were written under
`training/reproduction/` so the authors' supplied checkpoints remain untouched.

## Platform

- OS: Windows / PowerShell
- Python: 3.10.20 in `.venv`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB
- PyTorch: 2.11.0 + CUDA 12.8 build
- Test dataset: 3,730 simulated vessel images from the authors' HDF5 data

## Official-checkpoint sanity checks

`inferenceExperimental.py` completed with the supplied experimental-phantom
checkpoints and regenerated `experimentalOutput15dB.png`.

`inferenceSimulated.py` completed with the repository default `pSNRval=10`,
which generated measured non-inverse-crime data SNR around 14.46 dB and
therefore corresponds to the paper's 15 dB measurement-SNR setting.

| Method/checkpoint | Mean final pSNR (dB) |
|---|---:|
| L1 | 19.11 |
| TV | 27.26 |
| L1+TV | 24.88 |
| PP-MPI | 30.67 |
| DEQ-MPI, supplied pSNR-10 checkpoint | **32.19** |
| DEQ-MPI, supplied pSNR-15 checkpoint | 31.95 |

The paper reports `32.1 +/- 4.5 dB` for DEQ-MPI at 15 dB measurement SNR in
Table I. The supplied-checkpoint evaluation agrees with that value to the
shown precision.

## Full RDN pretraining

RDN image-prior pretraining completed for 200 epochs with the lightweight
configuration used by the supplied checkpoint: 12 features, 4 residual-dense
blocks, 4 layers per block, growth rate 12, fixed Gaussian noise std 0.1, and
image rescaling 0.5-1.5.

- final train loss: `0.023289`
- final train pSNR: `29.563950 dB`
- final validation loss: `0.023328`
- final validation pSNR: `29.558994 dB`
- checkpoint:
  `training/reproduction/denoiser/ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_nF12_nB4_lieb4_gr12_rMn0.5_1.0/epoch200END.pth`
- SHA256:
  `BF63ABC68155BA39C40C4694D845C5C0171078BE45E83C1ADB11285622F5B92B`

## Full learned-consistency pretraining

Learned-consistency pretraining completed for 200 epochs with the official
setting: 1-D consistency network, 8 features, 1 block, SVD 250, data pSNR 18,
system-matrix noise std `0.02`, normalization enabled, and L1 loss.

- final train loss: `0.000105`
- final train pSNR: `67.923256 dB`
- latest validation pSNR: `68.029541 dB` at epoch 198
- checkpoint:
  `training/reproduction/dcDenoiser/dcDenoiserPsi_1D_ds_lr_0.001_wd_0_bs_64_pSNR_18.0_fixNs_1_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_sN_0.02_ls_0/epoch200END.pth`
- SHA256:
  `B28549B0F23665187D6DD9926AA3AB71870925CA4FCB6B64073A66E9F8996183`

## Full DEQ-MPI joint training

The final DEQ-MPI training used the reproduced RDN and learned-consistency
checkpoints, pSNR 10, fixed noise std, SVD 250, image rescaling 0.5-1.5, batch
size 64, Adam learning rate `1e-3`, 200 global epochs, Anderson tolerance
`1e-4`, and at most 25 fixed-point iterations.

The long run was resumed across several segments after local process
interruptions:

- initial segment saved through `epoch20.pth`
- second segment resumed from `epoch20.pth`
- third segment resumed from `epoch70.pth`
- fourth segment resumed from `epoch90.pth`
- final segment resumed from `epoch150.pth` and used 5-epoch periodic saves

The resume path reloads the model weights and continues the global epoch
counter. The optimizer state is re-created, so this is a practical
reproduction workflow rather than a bit-exact continuation of a single
uninterrupted optimizer trajectory.

Final training metrics:

- final logged train loss at epoch 199: `0.026902`
- final logged train pSNR at epoch 199: `26.825563 dB`
- checkpoint:
  `training/reproduction/deqmpi/DeqMPI_1D_ds_lr_0.001_wd_0_bs_64_pSNR_10.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1/epoch200END.pth`
- SHA256:
  `40FA594874FA008687E204A0CBB8661B24015205CBE7C440DD6493281EBED38C`

## Final reproduced-checkpoint evaluation

`eval_repro_deqmpi.py` evaluated the reproduced `epoch200END.pth` on all
3,730 simulated test images with `pSNRval=10`, the inverse-crime-avoiding
forward model, SVD 220, and 25 fixed-point iterations.

- measured non-inverse-crime SNR: `14.419573 dB`
- measured inverse-crime SNR: `14.400216 dB`
- LS/SVD initialization mean pSNR: `12.334628 dB`
- reproduced DEQ-MPI mean pSNR: `32.193039 dB`
- reproduced DEQ-MPI pSNR std: `4.514024 dB`
- min/max reproduced DEQ-MPI pSNR: `21.761530 / 48.338905 dB`
- output JSON:
  `training/reproduction/deqmpi_eval_epoch200END.json`
- JSON SHA256:
  `2F07C380DBCBAAA7378A9C6027F3D8C9C4C5F1916E187FFF04FBC522D8A25F90`

This final reproduced checkpoint matches both the supplied official-checkpoint
evaluation (`32.19 dB`) and the paper's Table I value (`32.1 +/- 4.5 dB`) to
the reported precision.
