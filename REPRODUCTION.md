# DEQ-MPI reproduction log

This checkout reproduces the official DEQ-MPI implementation at commit
`f09607a4ac721da26643eecd1da208354a5a7c0c`.

## Local platform

- OS: Windows, PowerShell
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB
- Driver: 596.36
- Driver-reported CUDA capability: 13.2
- Python: 3.10.20 in `.venv`
- Runtime: modern CUDA 12.8 PyTorch build (the paper's CUDA 11.3 build does
  not support this GPU generation)

## Reproduction stages

1. Run the supplied experimental phantom through an official pretrained
   DEQ-MPI checkpoint.
2. Download the authors' MRA-derived HDF5 datasets and run simulated
   inference.
3. Verify the supplied simulated-checkpoint result at the paper's 15 dB
   measurement-SNR setting.
4. Pretrain the RDN image prior and learned-consistency network.
5. Train the complete implicit DEQ-MPI model and compare it with the supplied
   checkpoints.

## Environment setup

```powershell
uv python install 3.10
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe -r requirements-repro-cu128.txt
```

The original `requirements.txt` remains untouched for historical fidelity.
It specifies PyTorch 1.11/CUDA 11.3 and is unsuitable for the local RTX 5060.

## Paper-critical settings

- Image prior: RDN, 12 features, 4 residual-dense blocks, 12 layers/block.
- Learned consistency: one 1-D convolutional block, 8 features.
- Forward equilibrium solver during training: Anderson acceleration,
  tolerance `1e-4`, at most 25 iterations, `beta=2.0`.
- Initialization: truncated-SVD least-squares image; zero ADMM dual states.
- Training: Adam, learning rate `1e-3`, 200 epochs.

## Known code/paper discrepancy

`train_deqmpi.py` uses Anderson acceleration as described in the paper, but
`getModelForImplicitLD()` in `modelClasses.py` loads inference checkpoints with
25 plain fixed-point iterations. Both paths must be evaluated and labeled
separately instead of silently treating them as identical.

## Local compatibility patch

`train_ppmpi.py` now guards the StepLR interval with `max(1, epoch_nb // 5)`.
This only affects short smoke tests; the paper's 200-epoch interval remains
40 exactly as in the official implementation.

The script also accepts `--outputRoot` so local training cannot overwrite the
authors' supplied checkpoints. This changes only artifact placement.

The same isolated-output option and short-run scheduler guard are applied to
`train_dcDenoiser.py`. A precedence bug in the LC checkpoint condition was
fixed from `epoch + 1 % interval` to `(epoch + 1) % interval`; this restores
the intended periodic saves without changing optimization.

LC normalization formerly modified the output of `torch.linalg.norm` in
place. Modern PyTorch rejects that operation during autograd, so it now uses
`maximum`/`clamp_min` and an out-of-place multiplication. The epsilon-ball
projection is mathematically unchanged.

`train_deqmpi.py` also supports isolated `--outputRoot` artifacts and an
explicit `--detectAnomaly` debugging switch. Anomaly tracing is disabled by
default because it changes runtime substantially but not model mathematics.

For long DEQ-MPI runs, `train_deqmpi.py` additionally supports
`--resumeModel` and `--startEpoch`. These load a full model `state_dict` and
continue the global epoch counter, allowing interrupted runs such as
`epoch20.pth -> epoch200END.pth` to resume without overwriting the supplied
checkpoints. The optimizer state is re-created, so this is a practical
reproduction resume path rather than a bit-exact training-process snapshot.

The DEQ-MPI periodic checkpoint write now happens after the epoch's training
and optional validation have completed. The original placement was at the top
of the loop, which made names such as `epoch70.pth` slightly ahead of the
weights they contained and was awkward for interrupted long runs.

`eval_repro_deqmpi.py` is a lightweight final-check script for the reproduced
DEQ-MPI checkpoint. It evaluates only one checkpoint on the simulated test set,
using the same inverse-crime-avoiding setup as `inferenceSimulated.py`, and
writes a small JSON summary instead of the large `.mat` artifacts.
