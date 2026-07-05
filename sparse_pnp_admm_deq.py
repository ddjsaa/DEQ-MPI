"""Sparse-PnP-ADMM-DEQ fixed-point module for MPI reconstruction.

Example smoke usage:
  python eval_sparse_pnp_admm_deq.py --testLimit 32 --maxIter 12

The fixed-point state is packed as w = (x, z, s, u, v), where z is the
plug-and-play denoiser branch and s is the explicit l1 proximal branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn


def _as_shape(image_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(v) for v in image_shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"image_shape must be 2D or 3D, got {shape}")
    return shape


def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(threshold):
        threshold = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
    while threshold.ndim < x.ndim:
        threshold = threshold.unsqueeze(-1)
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


def _nonnegative_soft_threshold(x: torch.Tensor, threshold: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(threshold):
        threshold = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
    while threshold.ndim < x.ndim:
        threshold = threshold.unsqueeze(-1)
    return torch.relu(x - threshold)


def _row_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1)


@dataclass(frozen=True)
class SparsePnPADMMConfig:
    image_shape: tuple[int, ...]
    rho: float = 1.0
    eta: float = 1.0
    l1_lambda: float = 0.0
    non_negative: bool = True
    normalize_denoiser_input: bool = False
    denoiser_sigma: Optional[float] = None


class SparsePnPADMMFixedPoint(nn.Module):
    """One Sparse-PnP-ADMM fixed-point map T(w; y, A)."""

    def __init__(
        self,
        A: Optional[torch.Tensor],
        image_shape: Sequence[int],
        denoiser: Optional[nn.Module] = None,
        rho: float = 1.0,
        eta: float = 1.0,
        l1_lambda: float = 0.0,
        non_negative: bool = True,
        normalize_denoiser_input: bool = False,
        denoiser_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.config = SparsePnPADMMConfig(
            image_shape=_as_shape(image_shape),
            rho=float(rho),
            eta=float(eta),
            l1_lambda=float(l1_lambda),
            non_negative=bool(non_negative),
            normalize_denoiser_input=bool(normalize_denoiser_input),
            denoiser_sigma=None if denoiser_sigma is None else float(denoiser_sigma),
        )
        self.n_img = int(torch.tensor(self.config.image_shape).prod().item())
        self.denoiser = nn.Identity() if denoiser is None else denoiser
        self.register_buffer("A", None)
        self.register_buffer("x_update_matrix", None)
        if A is not None:
            self.set_system_matrix(A)

    @property
    def state_size(self) -> int:
        return 5 * self.n_img

    @property
    def rho(self) -> float:
        return self.config.rho

    @property
    def eta(self) -> float:
        return self.config.eta

    @property
    def l1_lambda(self) -> float:
        return self.config.l1_lambda

    def set_system_matrix(self, A: torch.Tensor) -> None:
        A = A.detach()
        if A.ndim != 2:
            raise ValueError(f"A must be [measurements, pixels], got {tuple(A.shape)}")
        if A.shape[1] != self.n_img:
            raise ValueError(f"A has {A.shape[1]} pixels, expected {self.n_img}")
        eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        mat = A.T @ A + (self.rho + self.eta) * eye
        self.A = A
        self.x_update_matrix = torch.linalg.inv(mat)

    def pack_state(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                x.reshape(x.shape[0], -1),
                z.reshape(z.shape[0], -1),
                s.reshape(s.shape[0], -1),
                u.reshape(u.shape[0], -1),
                v.reshape(v.shape[0], -1),
            ],
            dim=1,
        )

    def unpack_state(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if state.shape[1] != self.state_size:
            raise ValueError(f"state has {state.shape[1]} values, expected {self.state_size}")
        return tuple(state[:, i * self.n_img : (i + 1) * self.n_img] for i in range(5))

    def initial_state(self, batch_size: int, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x0 is None:
            if self.A is None:
                raise ValueError("initial_state needs x0 when no system matrix is registered")
            x = torch.zeros(batch_size, self.n_img, device=self.A.device, dtype=self.A.dtype)
        else:
            x = x0.reshape(batch_size, -1)
            if x.shape[1] != self.n_img:
                raise ValueError(f"x0 has {x.shape[1]} pixels, expected {self.n_img}")
            if self.config.non_negative:
                x = x.clamp_min(0.0)
        zeros = torch.zeros_like(x)
        return self.pack_state(x, x, x, zeros, zeros)

    def output_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.unpack_state(state)[0]

    def _parse_fixed_params(self, fixed_params: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(fixed_params, dict):
            y = fixed_params["y"]
            A = fixed_params.get("A", self.A)
        elif isinstance(fixed_params, (tuple, list)):
            y = fixed_params[0]
            A = fixed_params[1] if len(fixed_params) > 1 else self.A
        else:
            y = fixed_params
            A = self.A
        if A is None:
            raise ValueError("A must be registered in the module or passed in fixed_params")
        return y.reshape(y.shape[0], -1), A

    def _x_update(self, y: torch.Tensor, A: torch.Tensor, z: torch.Tensor, s: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        rhs = y @ A + self.rho * (z - u) + self.eta * (s - v)
        if A is self.A and self.x_update_matrix is not None:
            x = rhs @ self.x_update_matrix.T
        else:
            eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
            mat = A.T @ A + (self.rho + self.eta) * eye
            x = torch.linalg.solve(mat, rhs.T).T
        return x.clamp_min(0.0) if self.config.non_negative else x

    def _call_denoiser(self, image: torch.Tensor, sigma: Optional[torch.Tensor]) -> torch.Tensor:
        if sigma is None:
            return self.denoiser(image)
        try:
            return self.denoiser(image, sigma)
        except TypeError:
            return self.denoiser(image)

    def _denoise(self, x: torch.Tensor) -> torch.Tensor:
        image = x.reshape(x.shape[0], 1, *self.config.image_shape)
        sigma = None
        if self.config.denoiser_sigma is not None:
            sigma = torch.full((x.shape[0],), self.config.denoiser_sigma, device=x.device, dtype=x.dtype)

        if self.config.normalize_denoiser_input:
            dims = tuple(range(1, image.ndim))
            x_min = image.amin(dim=dims, keepdim=True)
            x_max = image.amax(dim=dims, keepdim=True)
            scale = (x_max - x_min).clamp_min(1e-8)
            denoised = self._call_denoiser((image - x_min) / scale, sigma)
            denoised = denoised.to(dtype=image.dtype) * scale + x_min
        else:
            denoised = self._call_denoiser(image, sigma).to(dtype=image.dtype)

        z = denoised.reshape_as(x)
        return z.clamp_min(0.0) if self.config.non_negative else z

    def _sparse_prox(self, x: torch.Tensor) -> torch.Tensor:
        threshold = self.l1_lambda / max(self.eta, 1e-12)
        if self.config.non_negative:
            return _nonnegative_soft_threshold(x, threshold)
        return _soft_threshold(x, threshold)

    def forward(self, state: torch.Tensor, fixed_params: Any) -> torch.Tensor:
        y, A = self._parse_fixed_params(fixed_params)
        x, z, s, u, v = self.unpack_state(state)

        x_next = self._x_update(y, A, z, s, u, v)
        z_next = self._denoise(x_next + u)
        s_next = self._sparse_prox(x_next + v)
        u_next = u + x_next - z_next
        v_next = v + x_next - s_next

        return self.pack_state(x_next, z_next, s_next, u_next, v_next)

    def transition_diagnostics(
        self,
        old_state: torch.Tensor,
        new_state: torch.Tensor,
        fixed_params: Any,
    ) -> dict[str, torch.Tensor]:
        y, A = self._parse_fixed_params(fixed_params)
        x, z, s, _, _ = self.unpack_state(new_state)
        delta = _row_norm(new_state - old_state)
        denom = _row_norm(old_state).clamp_min(1e-12)
        return {
            "fixed_point_abs": delta,
            "fixed_point_rel": delta / denom,
            "primal_z": _row_norm(x - z),
            "primal_s": _row_norm(x - s),
            "data_residual": _row_norm(x @ A.T - y),
        }

    def residuals(self, state: torch.Tensor, fixed_params: Any) -> dict[str, torch.Tensor]:
        next_state = self(state, fixed_params)
        return self.transition_diagnostics(state, next_state, fixed_params)

    def iterate(
        self,
        fixed_params: Any,
        state: torch.Tensor,
        max_iter: int,
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, float]]]:
        history: list[dict[str, float]] = []
        current = state
        for idx in range(int(max_iter)):
            nxt = self(current, fixed_params)
            if return_history:
                diag = self.transition_diagnostics(current, nxt, fixed_params)
                history.append(
                    {
                        "iter": idx + 1,
                        "fixed_point_abs_mean": float(diag["fixed_point_abs"].mean().detach().cpu()),
                        "fixed_point_rel_mean": float(diag["fixed_point_rel"].mean().detach().cpu()),
                        "primal_z_mean": float(diag["primal_z"].mean().detach().cpu()),
                        "primal_s_mean": float(diag["primal_s"].mean().detach().cpu()),
                        "data_residual_mean": float(diag["data_residual"].mean().detach().cpu()),
                    }
                )
            current = nxt
        if return_history:
            return current, history
        return current


class SparsePnPADMMDEQ(nn.Module):
    """Convenience wrapper around SparsePnPADMMFixedPoint.

    solver="finite" runs ordinary fixed-point iterations. solver="deq" wraps the
    map in the repository's DEQFixedPoint class for implicit differentiation.
    """

    def __init__(
        self,
        fixed_point: SparsePnPADMMFixedPoint,
        max_iter: int = 25,
        solver: str = "finite",
        deq_solver: str = "fixed",
        solver_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.max_iter = int(max_iter)
        self.solver = solver
        self.deq_solver = deq_solver
        self.solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
        self.deq_layer: Optional[nn.Module] = None
        if solver == "deq":
            object.__setattr__(self, "fixed_point", fixed_point)
            from modelClasses import DEQFixedPoint, anderson, fixedPointTekrarlar, regularFixed

            solver_map = {
                "fixed": fixedPointTekrarlar,
                "regular": regularFixed,
                "anderson": anderson,
            }
            if deq_solver not in solver_map:
                raise ValueError(f"unknown deq_solver: {deq_solver}")
            kwargs = {"max_iter": self.max_iter}
            kwargs.update(self.solver_kwargs)
            self.deq_layer = DEQFixedPoint(self.fixed_point, solver_map[deq_solver], **kwargs)
        elif solver == "finite":
            self.fixed_point = fixed_point
        else:
            raise ValueError(f"unknown solver: {solver}")

    def forward(
        self,
        fixed_params: Any,
        x0: Optional[torch.Tensor] = None,
        return_state: bool = False,
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        y, _ = self.fixed_point._parse_fixed_params(fixed_params)
        state0 = self.fixed_point.initial_state(y.shape[0], x0=x0)
        history: list[dict[str, float]] = []
        if self.solver == "finite":
            if return_history:
                state, history = self.fixed_point.iterate(fixed_params, state0, self.max_iter, return_history=True)
            else:
                state = self.fixed_point.iterate(fixed_params, state0, self.max_iter, return_history=False)
        else:
            if return_history:
                raise ValueError("return_history is available only with solver='finite'")
            assert self.deq_layer is not None
            state = self.deq_layer(state0, fixed_params)

        x = self.fixed_point.output_from_state(state)
        if return_state and return_history:
            return x, state, history
        if return_state:
            return x, state
        if return_history:
            return x, history
        return x


def mean_tensor_diagnostics(diagnostics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key + "_mean": float(value.mean().detach().cpu()) for key, value in diagnostics.items()}
