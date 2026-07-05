import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    shape = tuple(int(v) for v in shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"image shape must be 2D or 3D, got {shape}")
    return shape


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(threshold):
        threshold = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
    while threshold.ndim < x.ndim:
        threshold = threshold.unsqueeze(-1)
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


def normalize_for_denoiser(x: torch.Tensor, eps: float = 1e-8):
    x_min = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
    x_max = x.amax(dim=tuple(range(1, x.ndim)), keepdim=True)
    scale = torch.clamp(x_max - x_min, min=eps)
    return (x - x_min) / scale, x_min, scale


class GaussianDenoiser(nn.Module):
    """Small deterministic denoiser for smoke tests when DRUNet weights are absent."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.reshape(1, 1, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.conv2d(x, self.kernel.to(dtype=x.dtype, device=x.device), padding=self.kernel.shape[-1] // 2)


class DPIRDrunetDenoiser(nn.Module):
    """Wrapper for the official DPIR/KAIR DRUNet gray denoiser.

    Expected setup:
      git clone https://github.com/cszn/DPIR external/DPIR
      download drunet_gray.pth from the DPIR/KAIR model zoo

    The underlying DRUNet accepts two channels: the image and a noise-level map.
    """

    def __init__(self, dpir_root: str | Path, weights: str | Path, device: torch.device):
        super().__init__()
        dpir_root = Path(dpir_root)
        weights = Path(weights)
        if not dpir_root.exists():
            raise FileNotFoundError(f"DPIR root not found: {dpir_root}")
        if not weights.exists():
            raise FileNotFoundError(f"DRUNet weights not found: {weights}")
        sys.path.insert(0, str(dpir_root))
        try:
            from models.network_unet import UNetRes
        except Exception as exc:
            raise ImportError(
                "Could not import DPIR models.network_unet.UNetRes. "
                "Pass --dpirRoot pointing at a cloned https://github.com/cszn/DPIR repo."
            ) from exc

        self.model = UNetRes(
            in_nc=2,
            out_nc=1,
            nc=[64, 128, 256, 512],
            nb=4,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        ).to(device)
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and "params" in state:
            state = state["params"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        if sigma is None:
            sigma = torch.full((x.shape[0], 1, 1, 1), 15.0 / 255.0, device=x.device, dtype=x.dtype)
        if sigma.ndim == 1:
            sigma = sigma[:, None, None, None]
        sigma = sigma.to(device=x.device, dtype=x.dtype).clamp(0.0, 50.0 / 255.0)
        sigma_map = sigma.expand(x.shape[0], 1, x.shape[-2], x.shape[-1])
        model_in = torch.cat([x, sigma_map], dim=1)
        height, width = model_in.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            model_in = F.pad(model_in, (0, pad_w, 0, pad_h), mode="replicate")
        out = self.model(model_in).clamp(0.0, 1.0)
        return out[..., :height, :width]


class SlicewiseDenoiser(nn.Module):
    def __init__(self, base_denoiser: nn.Module, image_shape: Sequence[int], axes: Sequence[int] = (0, 1, 2)):
        super().__init__()
        self.base_denoiser = base_denoiser
        self.image_shape = _as_shape(image_shape)
        self.axes = tuple(int(a) for a in axes)

    @torch.no_grad()
    def forward(self, x_flat: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if len(self.image_shape) == 2:
            x = x_flat.reshape(-1, 1, *self.image_shape)
            x_norm, x_min, scale = normalize_for_denoiser(x)
            sigma_norm = (sigma / scale.flatten(start_dim=1).mean(dim=1).clamp_min(1e-8)).clamp(0.0, 50.0 / 255.0)
            out = self.base_denoiser(x_norm, sigma_norm).to(dtype=x.dtype)
            return (out * scale + x_min).reshape_as(x_flat)

        volume = x_flat.reshape(-1, *self.image_shape)
        outputs = []
        for axis in self.axes:
            moved = volume.movedim(axis + 1, 1)
            num_slices = moved.shape[1]
            slices = moved.reshape(-1, 1, moved.shape[-2], moved.shape[-1])
            slices_norm, x_min, scale = normalize_for_denoiser(slices)
            sigma_rep = sigma.repeat_interleave(num_slices)
            sigma_norm = (sigma_rep / scale.flatten(start_dim=1).mean(dim=1).clamp_min(1e-8)).clamp(0.0, 50.0 / 255.0)
            den = self.base_denoiser(slices_norm, sigma_norm).to(dtype=slices.dtype)
            den = (den * scale + x_min).reshape(moved.shape)
            outputs.append(den.movedim(1, axis + 1))
        return torch.stack(outputs, dim=0).mean(dim=0).reshape_as(x_flat)


class ZeroShotL1PnP:
    """ZeroShot-l1-PnP reconstruction following Algorithm 1 of Gapyak et al.

    This implementation supports the 2D DEQ-MPI reproduction data and 3D OpenMPI
    volumes. The denoiser is injected as a module, so a real DRUNet can be used
    when DPIR weights are available, while GaussianDenoiser keeps tests runnable.
    """

    def __init__(
        self,
        A: torch.Tensor,
        image_shape: Sequence[int],
        denoiser: nn.Module,
        n_iter: int = 7,
        mu0: float = 2e5,
        alpha: Optional[float] = None,
        cg_max_iter: int = 1000,
        cg_tol: float = 1e-8,
        direct_solve: bool = True,
        non_negative: bool = True,
    ):
        self.A = A
        self.image_shape = _as_shape(image_shape)
        self.denoiser = SlicewiseDenoiser(denoiser, self.image_shape).to(A.device)
        self.n_iter = int(n_iter)
        self.mu0 = float(mu0)
        self.alpha = 0.005 * self.mu0 if alpha is None else float(alpha)
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.direct_solve = bool(direct_solve)
        self.non_negative = bool(non_negative)
        self.AT = A.T.conj()
        self.ATA = self.AT @ A
        self.eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)

    def _solve(self, rhs: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        mu_scalar = float(mu.detach().mean().cpu())
        mat = self.ATA + mu_scalar * self.eye
        if self.direct_solve:
            return torch.linalg.solve(mat, rhs.T).T
        return torch.stack([self._cg_solve(mat, row) for row in rhs], dim=0)

    def _cg_solve(self, mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(rhs)
        r = rhs - mat @ x
        p = r.clone()
        rs_old = torch.dot(r, r)
        for _ in range(self.cg_max_iter):
            Ap = mat @ p
            alpha = rs_old / torch.clamp(torch.dot(p, Ap), min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < self.cg_tol:
                break
            p = r + (rs_new / torch.clamp(rs_old, min=1e-30)) * p
            rs_old = rs_new
        return x

    @torch.no_grad()
    def __call__(self, y: torch.Tensor, return_history: bool = False, x0: Optional[torch.Tensor] = None):
        y = y.reshape(y.shape[0], -1).to(device=self.A.device, dtype=self.A.dtype)
        batch, n_vox = y.shape[0], self.A.shape[1]
        if x0 is None:
            u2 = torch.zeros(batch, n_vox, device=y.device, dtype=y.dtype)
        else:
            u2 = x0.reshape(batch, -1).to(device=y.device, dtype=y.dtype)
            if u2.shape[1] != n_vox:
                raise ValueError(f"x0 has {u2.shape[1]} voxels, expected {n_vox}")
        u3 = u2.clone()
        mu = torch.as_tensor(self.mu0, device=y.device, dtype=y.dtype)
        lam = None
        history = []

        ATy = y @ self.A
        for k in range(self.n_iter):
            center = 0.5 * (u2 + u3)
            u1 = self._solve(ATy + mu * center, mu)
            sigma = torch.sqrt(torch.var(u1, dim=1, unbiased=False).clamp_min(1e-12))
            sigma_scalar = sigma.mean()
            if lam is None:
                lam = mu * sigma_scalar.square()

            u2 = self.denoiser(u1, sigma)
            if self.non_negative:
                u2 = u2.clamp_min(0.0)

            threshold = self.alpha / mu
            u3 = soft_threshold(u1, threshold)
            if self.non_negative:
                u3 = u3.clamp_min(0.0)

            mu = lam / sigma_scalar.square().clamp_min(1e-12)
            if return_history:
                data_res = torch.linalg.norm(y - u2 @ self.A.T, dim=1).mean()
                history.append(
                    {
                        "iter": k + 1,
                        "mu": float(mu.detach().cpu()),
                        "sigma": float(sigma_scalar.detach().cpu()),
                        "data_residual": float(data_res.detach().cpu()),
                    }
                )

        if return_history:
            return u2, history
        return u2


def build_denoiser(args, device: torch.device) -> nn.Module:
    if args.denoiser == "gaussian":
        return GaussianDenoiser(kernel_size=args.gaussianKernel, sigma=args.gaussianSigma).to(device).eval()
    if args.denoiser == "drunet":
        return DPIRDrunetDenoiser(args.dpirRoot, args.drunetWeights, device).eval()
    raise ValueError(f"unknown denoiser: {args.denoiser}")
