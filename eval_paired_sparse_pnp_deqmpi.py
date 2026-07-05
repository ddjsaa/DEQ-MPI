"""Paired Sparse-PnP-ADMM-DEQ vs DEQ-MPI evaluation.

This evaluator generates one deterministic target/noisy-measurement problem and
runs both checkpoints on that exact problem. It avoids comparing per-sample CSVs
produced by separate scripts, whose random intensity scaling/noise paths can
differ even when they use the same seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from data import MRAdatasetH5NoScale, loadMtxExp
from diagnose_sparse_pnp_admm_deq_gap import make_bins, pearson, stats, values, write_csv
from modelClasses import DEQFixedPoint, fixedPointTekrarlar, rdnLDFixedPt
from reconAlgos import ADMMfncs
from sparse_pnp_admm_deq import SparsePnPADMMDEQ, SparsePnPADMMFixedPoint
from trainerClasses import getNoisyData, transformDataset
from train_sparse_pnp_admm_deq import make_denoiser


DEFAULT_SPARSE_RUN = Path(
    "training/reproduction/sparse_pnp_admm_deq/"
    "deq_anderson_rdn_full_e3_it12_eta0p10"
)
DEFAULT_DEQ_CHECKPOINT = Path(
    "training/reproduction/deqmpi/"
    "DeqMPI_1D_ds_lr_0.001_wd_0_bs_64_pSNR_10.0_fixNs_1_Nit_5_"
    "nF12_nB4_lieb4_gr12_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_"
    "LnF_8_LnB_1_nN_1/epoch200END.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paired Sparse-PnP-DEQ vs DEQ-MPI evaluation")
    parser.add_argument("--sparseRunDir", default=str(DEFAULT_SPARSE_RUN))
    parser.add_argument("--sparseCheckpoint", default="")
    parser.add_argument("--deqCheckpoint", default=str(DEFAULT_DEQ_CHECKPOINT))
    parser.add_argument("--mode", choices=["direct", "canonical"], default="direct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--testPath", default="datasets/testPatches.h5")
    parser.add_argument("--testLimit", type=int, default=0)
    parser.add_argument("--batchSizeSparse", type=int, default=128)
    parser.add_argument("--batchSizeDeq", type=int, default=256)
    parser.add_argument("--nbOfSingulars", type=int, default=0)
    parser.add_argument("--pSNRval", type=float, default=-1.0)
    parser.add_argument("--rescaleMin", type=float, default=-1.0)
    parser.add_argument("--rescaleMax", type=float, default=-1.0)
    parser.add_argument("--sparseMaxIterOverride", type=int, default=0)
    parser.add_argument("--sparseSolverOverride", choices=["", "finite", "deq"], default="")
    parser.add_argument(
        "--sparseDeqSolverOverride",
        choices=["", "fixed", "regular", "anderson"],
        default="",
    )
    parser.add_argument("--sparseRhoOverride", type=float, default=-1.0)
    parser.add_argument("--sparseEtaOverride", type=float, default=-1.0)
    parser.add_argument("--sparseL1LambdaOverride", type=float, default=-1.0)
    parser.add_argument(
        "--sparseDenoiserOverride",
        choices=["", "identity", "gaussian", "rdn"],
        default="",
    )
    parser.add_argument(
        "--skipSparseCheckpoint",
        action="store_true",
        help="Evaluate the Sparse model from its configured/preloaded denoiser without loading a Sparse checkpoint.",
    )
    parser.add_argument("--maxIterDeq", type=int, default=25)
    parser.add_argument("--outDir", default=str(DEFAULT_SPARSE_RUN / "paired_evaluations"))
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def setup_device(gpu: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because repository matrix utilities use CUDA tensors.")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def load_run_args(run_dir: Path):
    with (run_dir / "args.json").open("r", encoding="utf-8") as f:
        return SimpleNamespace(**json.load(f))


def apply_sparse_overrides(run_args, args):
    if args.sparseMaxIterOverride > 0:
        run_args.maxIter = args.sparseMaxIterOverride
    if args.sparseSolverOverride:
        run_args.solver = args.sparseSolverOverride
    if args.sparseDeqSolverOverride:
        run_args.deqSolver = args.sparseDeqSolverOverride
    if args.sparseRhoOverride >= 0:
        run_args.rho = args.sparseRhoOverride
    if args.sparseEtaOverride >= 0:
        run_args.eta = args.sparseEtaOverride
    if args.sparseL1LambdaOverride >= 0:
        run_args.l1Lambda = args.sparseL1LambdaOverride
    if args.sparseDenoiserOverride:
        run_args.denoiser = args.sparseDenoiserOverride
    return run_args


def sparse_checkpoint_path(args, run_dir: Path):
    if args.sparseCheckpoint:
        return Path(args.sparseCheckpoint)
    best = run_dir / "best.pth"
    if best.exists():
        return best
    finals = sorted(run_dir.glob("epoch*END.pth"))
    if not finals:
        raise FileNotFoundError(f"No Sparse checkpoint found in {run_dir}")
    return finals[-1]


def build_problem(args, run_args, device):
    n1, n2 = 26, 13
    img_size = [n1, n2]
    n_img = n1 * n2

    p_snr = run_args.pSNRval if args.pSNRval < 0 else args.pSNRval
    rescale_min = run_args.rescaleMin if args.rescaleMin < 0 else args.rescaleMin
    rescale_max = run_args.rescaleMax if args.rescaleMax < 0 else args.rescaleMax
    nb_singulars = run_args.nbOfSingulars if args.nbOfSingulars <= 0 else args.nbOfSingulars

    sys_mtx_ref = loadMtxExp("inhouseData/expMatinHouse").reshape(-1, n_img).to(device)
    if args.mode == "canonical":
        interpolater = loadmat("interpExp2.mat")["interpolater"]
        sys_mtx_hr_int2 = sys_mtx_ref @ torch.from_numpy(interpolater).float().to(device)
        sys_mtx_recon = F.avg_pool2d(
            sys_mtx_hr_int2.reshape(sys_mtx_ref.shape[0], 2 * n1, 2 * n2), 2
        ).reshape(sys_mtx_ref.shape[0], -1)
        sys_mtx_measure = sys_mtx_ref
    else:
        sys_mtx_recon = sys_mtx_ref
        sys_mtx_measure = sys_mtx_ref

    u, s, vh = torch.linalg.svd(sys_mtx_recon, full_matrices=False)
    u = u[:, :nb_singulars]
    s = s[:nb_singulars]
    v = vh[:nb_singulars, :].T
    a_reduced = torch.diag(s) @ v.T

    data_hr = MRAdatasetH5NoScale(
        args.testPath, prefetch=True, dim=2, device=torch.device("cpu")
    ).data
    if args.testLimit and args.testLimit > 0:
        data_hr = data_hr[: args.testLimit].clone()
    data_hr = data_hr.to(device).float()
    target_img = transformDataset(data_hr, img_size, [rescale_min, rescale_max], [0, 0])
    target = target_img.reshape(target_img.shape[0], -1)

    std_val = 0.41 * 10 ** (-p_snr / 20)
    noiseless_measure = F.linear(target_img.reshape(target_img.shape[0], 1, -1), sys_mtx_measure)
    noisy_full = getNoisyData(target_img, std_val, sys_mtx_measure)
    datat_c_full = noisy_full.reshape(target_img.shape[0], -1)
    y_reduced = datat_c_full @ u
    x0 = F.linear(y_reduced / (s + 1e-4), v).reshape(target_img.shape[0], -1)

    noise = noisy_full - noiseless_measure
    measured_snr = float(20 * torch.log10(torch.norm(noiseless_measure) / torch.norm(noise)))
    epsilon = float(std_val * (noiseless_measure.shape[2]) ** 0.5)

    return {
        "n1": n1,
        "n2": n2,
        "n_img": n_img,
        "img_size": img_size,
        "target": target,
        "sys_mtx_recon": sys_mtx_recon,
        "sys_mtx_measure": sys_mtx_measure,
        "u": u,
        "s": s,
        "v": v,
        "a_reduced": a_reduced,
        "datat_c_full": datat_c_full,
        "y_reduced": y_reduced,
        "x0": x0,
        "epsilon": epsilon,
        "pSNRval": p_snr,
        "rescaleMin": rescale_min,
        "rescaleMax": rescale_max,
        "nbOfSingulars": nb_singulars,
        "measured_snr_db": measured_snr,
    }


def build_sparse_model(run_args, problem, device):
    denoiser = make_denoiser(run_args, device)
    fixed_point = SparsePnPADMMFixedPoint(
        A=problem["a_reduced"],
        image_shape=problem["img_size"],
        denoiser=denoiser,
        rho=run_args.rho,
        eta=run_args.eta,
        l1_lambda=run_args.l1Lambda,
        non_negative=not run_args.allowNegative,
    ).to(device)
    model = SparsePnPADMMDEQ(
        fixed_point,
        max_iter=run_args.maxIter,
        solver=run_args.solver,
        deq_solver=run_args.deqSolver,
        solver_kwargs={"tol": 1e-4, "beta": 1.0}
        if run_args.deqSolver in ("regular", "anderson")
        else None,
    ).to(device)
    return model, fixed_point


def load_sparse_weights(model, checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device)
    target_keys = set(model.state_dict().keys())
    removed = {}
    remapped = {}
    load_state = {}
    system_buffer_suffixes = (".A", ".x_update_matrix")
    system_buffer_names = {"A", "x_update_matrix"}

    for key, value in state.items():
        if key in system_buffer_names or key.endswith(system_buffer_suffixes):
            removed[key] = tuple(value.shape) if hasattr(value, "shape") else str(type(value))
            continue

        target_key = key
        if key not in target_keys and key.startswith("deq_layer.f."):
            candidate = "fixed_point." + key[len("deq_layer.f.") :]
            if candidate in target_keys:
                target_key = candidate
        elif key not in target_keys and key.startswith("fixed_point."):
            candidate = "deq_layer.f." + key[len("fixed_point.") :]
            if candidate in target_keys:
                target_key = candidate

        if target_key in target_keys:
            load_state[target_key] = value
            if target_key != key:
                remapped[key] = target_key

    result = model.load_state_dict(load_state, strict=False)
    return {
        "removed_system_buffers": removed,
        "remapped_keys": remapped,
        "skipped_unmapped_keys": [
            key
            for key in state
            if key not in removed
            and key not in load_state
            and not (
                key.startswith("deq_layer.f.")
                and "fixed_point." + key[len("deq_layer.f.") :] in load_state
            )
            and not (
                key.startswith("fixed_point.")
                and "deq_layer.f." + key[len("fixed_point.") :] in load_state
            )
        ],
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def run_sparse(model, problem, batch_size: int):
    y_reduced = problem["y_reduced"]
    x0 = problem["x0"]
    pred = torch.zeros_like(x0)
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, y_reduced.shape[0], batch_size):
            end = min(start + batch_size, y_reduced.shape[0])
            pred[start:end] = model(y_reduced[start:end], x0=x0[start:end])
    return pred, time.time() - start_time


def run_deq_mpi(args, problem, device, checkpoint: Path):
    n1, n2, n_img = problem["n1"], problem["n2"], problem["n_img"]
    datat_c_full = problem["datat_c_full"]
    val_inp = problem["x0"].reshape(-1, 1, n1, n2)

    at_c = problem["sys_mtx_recon"].reshape(-1, n_img)
    admm = ADMMfncs(at_c, args.maxIterDeq, 100, problem["img_size"], diagnose=False)
    madmm = admm.MtC

    model_core = rdnLDFixedPt(
        1,
        12,
        4,
        4,
        12,
        8,
        1,
        1,
        bias=True,
        numDim=2,
        consistencyDim=1,
    ).to(device)
    model = DEQFixedPoint(model_core, fixedPointTekrarlar, max_iter=args.maxIterDeq).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    n_batch = datat_c_full.shape[0]
    d0 = torch.zeros_like(val_inp).reshape(n_batch, -1)
    d2 = torch.zeros_like(datat_c_full)
    model_in = torch.cat((val_inp.reshape(n_batch, -1), d0, d2), dim=1)
    pred = torch.zeros_like(problem["target"]).reshape(-1, 1, n1, n2)

    start_time = time.time()
    with torch.no_grad():
        for start in range(0, n_batch, args.batchSizeDeq):
            end = min(start + args.batchSizeDeq, n_batch)
            batch_shape = (end - start, 1, n1, n2)
            fixed_pts = (
                datat_c_full[start:end],
                at_c,
                madmm,
                problem["epsilon"],
                batch_shape,
            )
            pred[start:end] = model(model_in[start:end], fixed_pts)[:, :n_img].reshape(
                batch_shape
            )
    return pred.reshape(n_batch, -1), time.time() - start_time


def psnr_values(target, pred):
    n_pix = target.shape[1]
    err = torch.linalg.norm(target - pred, dim=1).clamp_min(1e-12)
    peak = target.max().clamp_min(1e-12)
    return 10 * torch.log10(n_pix * peak.square() / err.square())


def nrmse_values(target, pred):
    err = torch.linalg.norm(target - pred, dim=1).clamp_min(1e-12)
    target_norm = torch.linalg.norm(target, dim=1).clamp_min(1e-12)
    return err / target_norm


def residual_values(pred, y_reduced, a_reduced):
    return torch.linalg.norm(y_reduced - pred @ a_reduced.T, dim=1)


def tensor_cpu_np(x):
    return x.detach().cpu().numpy()


def build_rows(problem, sparse_pred, deq_pred):
    target = problem["target"]
    x0 = problem["x0"]
    y_reduced = problem["y_reduced"]
    a_reduced = problem["a_reduced"]

    sparse_psnr = tensor_cpu_np(psnr_values(target, sparse_pred))
    deq_psnr = tensor_cpu_np(psnr_values(target, deq_pred))
    ls_psnr = tensor_cpu_np(psnr_values(target, x0))
    sparse_nrmse = tensor_cpu_np(nrmse_values(target, sparse_pred))
    deq_nrmse = tensor_cpu_np(nrmse_values(target, deq_pred))
    sparse_resid = tensor_cpu_np(residual_values(sparse_pred, y_reduced, a_reduced))
    deq_resid = tensor_cpu_np(residual_values(deq_pred, y_reduced, a_reduced))

    target_np = tensor_cpu_np(target)
    sparse_np = tensor_cpu_np(sparse_pred)
    deq_np = tensor_cpu_np(deq_pred)
    rows = []
    for idx in range(target.shape[0]):
        target_row = target_np[idx]
        sparse_row = sparse_np[idx]
        deq_row = deq_np[idx]
        row = {
            "sample_index": idx,
            "sparse_psnr_db": float(sparse_psnr[idx]),
            "deq_mpi_psnr_db": float(deq_psnr[idx]),
            "ls_svd_psnr_db": float(ls_psnr[idx]),
            "delta_sparse_minus_deq_mpi_db": float(sparse_psnr[idx] - deq_psnr[idx]),
            "gap_deq_mpi_minus_sparse_db": float(deq_psnr[idx] - sparse_psnr[idx]),
            "delta_sparse_minus_ls_svd_db": float(sparse_psnr[idx] - ls_psnr[idx]),
            "sparse_nrmse": float(sparse_nrmse[idx]),
            "deq_mpi_nrmse": float(deq_nrmse[idx]),
            "sparse_data_residual_norm": float(sparse_resid[idx]),
            "deq_mpi_data_residual_norm": float(deq_resid[idx]),
            "target_norm": float(np.linalg.norm(target_row)),
            "target_sum": float(target_row.sum()),
            "target_max_value": float(target_row.max()),
            "target_nonzero_ratio": float(np.mean(np.abs(target_row) > 1e-8)),
            "sparse_min_value": float(sparse_row.min()),
            "sparse_max_value": float(sparse_row.max()),
            "deq_mpi_min_value": float(deq_row.min()),
            "deq_mpi_max_value": float(deq_row.max()),
        }
        rows.append(row)
    return rows


def correlation_summary(rows, target_field, candidate_fields):
    target = values(rows, target_field)
    out = {}
    for field in candidate_fields:
        if field not in rows[0]:
            continue
        corr = pearson(values(rows, field), target)
        if corr is not None:
            out[field] = corr
    return dict(sorted(out.items(), key=lambda item: abs(item[1]), reverse=True))


def tail_summary(rows, fractions=(0.05, 0.10, 0.20)):
    ordered = sorted(rows, key=lambda row: row["delta_sparse_minus_deq_mpi_db"])
    out = {}
    for frac in fractions:
        n = max(1, int(np.ceil(len(rows) * frac)))
        worst = ordered[:n]
        best = ordered[-n:]
        out[f"worst_{int(frac * 100)}pct"] = {
            "count": n,
            "mean_delta_sparse_minus_deq_mpi_db": float(
                values(worst, "delta_sparse_minus_deq_mpi_db").mean()
            ),
            "mean_sparse_psnr_db": float(values(worst, "sparse_psnr_db").mean()),
            "mean_deq_mpi_psnr_db": float(values(worst, "deq_mpi_psnr_db").mean()),
            "mean_target_max_value": float(values(worst, "target_max_value").mean()),
            "mean_target_sum": float(values(worst, "target_sum").mean()),
        }
        out[f"best_{int(frac * 100)}pct"] = {
            "count": n,
            "mean_delta_sparse_minus_deq_mpi_db": float(
                values(best, "delta_sparse_minus_deq_mpi_db").mean()
            ),
            "mean_sparse_psnr_db": float(values(best, "sparse_psnr_db").mean()),
            "mean_deq_mpi_psnr_db": float(values(best, "deq_mpi_psnr_db").mean()),
            "mean_target_max_value": float(values(best, "target_max_value").mean()),
            "mean_target_sum": float(values(best, "target_sum").mean()),
        }
    return out


def write_report(path: Path, summary: dict):
    delta = summary["metrics"]["delta_sparse_minus_deq_mpi_db"]
    sparse = summary["metrics"]["sparse_psnr_db"]
    deq = summary["metrics"]["deq_mpi_psnr_db"]
    win = summary["win_rates"]["delta_gt_0"]
    tail = summary["tail"]
    correlations = summary["correlations"]["delta_sparse_minus_deq_mpi_db"]
    lines = [
        f"# Paired Evaluation: {summary['mode']}",
        "",
        f"- Samples: `{summary['num_samples']}`",
        f"- Sparse mean PSNR: `{sparse['mean']:.4f} dB`",
        f"- DEQ-MPI mean PSNR: `{deq['mean']:.4f} dB`",
        (
            f"- Delta Sparse minus DEQ-MPI: `{delta['mean']:.4f} dB` "
            f"(median `{delta['median']:.4f}`, q05/q95 `{delta['q05']:.4f}/{delta['q95']:.4f}`)"
        ),
        f"- Sparse win rate: `{100 * win:.2f}%`",
        (
            f"- Worst 5% delta: "
            f"`{tail['worst_5pct']['mean_delta_sparse_minus_deq_mpi_db']:.4f} dB`; "
            f"best 5% delta: `{tail['best_5pct']['mean_delta_sparse_minus_deq_mpi_db']:.4f} dB`"
        ),
        "",
        "## Correlations With Delta",
    ]
    for field, corr in list(correlations.items())[:8]:
        lines.append(f"- `{field}`: `{corr:.4f}`")
    lines.extend(["", "## Outputs"])
    for name, out_path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{out_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.sparseRunDir)
    sparse_ckpt = None if args.skipSparseCheckpoint else sparse_checkpoint_path(args, run_dir)
    deq_ckpt = Path(args.deqCheckpoint)
    run_args = apply_sparse_overrides(load_run_args(run_dir), args)

    device = setup_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    problem = build_problem(args, run_args, device)

    sparse_model, sparse_fixed_point = build_sparse_model(run_args, problem, device)
    if sparse_ckpt is None:
        sparse_load = {
            "checkpoint_loaded": False,
            "removed_system_buffers": {},
            "remapped_keys": {},
            "skipped_unmapped_keys": [],
            "missing_keys": [],
            "unexpected_keys": [],
        }
    else:
        sparse_load = load_sparse_weights(sparse_model, sparse_ckpt, device)
    sparse_model.eval()
    for param in sparse_model.parameters():
        param.requires_grad = False

    sparse_pred, sparse_elapsed = run_sparse(sparse_model, problem, args.batchSizeSparse)
    with torch.no_grad():
        _, first_state = sparse_model(
            problem["y_reduced"][:1],
            x0=problem["x0"][:1],
            return_state=True,
        )
        sparse_probe = sparse_fixed_point.residuals(first_state, problem["y_reduced"][:1])

    deq_pred, deq_elapsed = run_deq_mpi(args, problem, device, deq_ckpt)

    rows = build_rows(problem, sparse_pred, deq_pred)
    bin_fields = [
        "ls_svd_psnr_db",
        "deq_mpi_psnr_db",
        "sparse_psnr_db",
        "target_max_value",
        "target_sum",
        "target_norm",
        "sparse_max_value",
        "sparse_data_residual_norm",
    ]
    bin_rows = make_bins(rows, bin_fields)

    prefix = args.prefix or f"paired_{args.mode}_seed{args.seed}"
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sample_metrics_csv": str(out_dir / f"{prefix}_sample_metrics.csv"),
        "bin_summary_csv": str(out_dir / f"{prefix}_bin_summary.csv"),
        "summary_json": str(out_dir / f"{prefix}_summary.json"),
        "report_md": str(out_dir / f"{prefix}_report.md"),
    }

    metric_fields = [
        "sparse_psnr_db",
        "deq_mpi_psnr_db",
        "ls_svd_psnr_db",
        "delta_sparse_minus_deq_mpi_db",
        "gap_deq_mpi_minus_sparse_db",
        "sparse_nrmse",
        "deq_mpi_nrmse",
        "sparse_data_residual_norm",
        "deq_mpi_data_residual_norm",
        "target_max_value",
        "target_sum",
        "target_norm",
    ]
    metrics = {field: stats(values(rows, field)) for field in metric_fields}
    delta = values(rows, "delta_sparse_minus_deq_mpi_db")
    candidate_fields = [
        "sparse_psnr_db",
        "deq_mpi_psnr_db",
        "ls_svd_psnr_db",
        "sparse_nrmse",
        "deq_mpi_nrmse",
        "sparse_data_residual_norm",
        "deq_mpi_data_residual_norm",
        "target_max_value",
        "target_sum",
        "target_norm",
        "target_nonzero_ratio",
        "sparse_min_value",
        "sparse_max_value",
        "deq_mpi_min_value",
        "deq_mpi_max_value",
    ]
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "num_samples": len(rows),
        "sparse_checkpoint": None if sparse_ckpt is None else str(sparse_ckpt),
        "deq_checkpoint": str(deq_ckpt),
        "sparse_inference": {
            "solver": run_args.solver,
            "deqSolver": run_args.deqSolver if run_args.solver == "deq" else None,
            "maxIter": run_args.maxIter,
            "rho": run_args.rho,
            "eta": run_args.eta,
            "l1Lambda": run_args.l1Lambda,
        },
        "sparse_load": sparse_load,
        "problem": {
            "testPath": args.testPath,
            "pSNRval": problem["pSNRval"],
            "measured_snr_db": problem["measured_snr_db"],
            "rescaleMin": problem["rescaleMin"],
            "rescaleMax": problem["rescaleMax"],
            "nbOfSingulars": problem["nbOfSingulars"],
            "epsilon": problem["epsilon"],
        },
        "timing": {
            "sparse_elapsed_seconds": sparse_elapsed,
            "deq_mpi_elapsed_seconds": deq_elapsed,
        },
        "sparse_first_sample_probe": {
            key + "_mean": float(value.mean().detach().cpu()) for key, value in sparse_probe.items()
        },
        "metrics": metrics,
        "win_rates": {
            "delta_gt_0": float(np.mean(delta > 0.0)),
            "delta_gt_0p1": float(np.mean(delta > 0.1)),
            "delta_gt_1p0": float(np.mean(delta > 1.0)),
            "delta_lt_minus_0p1": float(np.mean(delta < -0.1)),
            "delta_lt_minus_1p0": float(np.mean(delta < -1.0)),
            "abs_delta_lte_0p1": float(np.mean(np.abs(delta) <= 0.1)),
            "abs_delta_lte_0p5": float(np.mean(np.abs(delta) <= 0.5)),
        },
        "tail": tail_summary(rows),
        "correlations": {
            "delta_sparse_minus_deq_mpi_db": correlation_summary(
                rows, "delta_sparse_minus_deq_mpi_db", candidate_fields
            )
        },
        "outputs": outputs,
    }

    write_csv(Path(outputs["sample_metrics_csv"]), rows)
    write_csv(Path(outputs["bin_summary_csv"]), bin_rows)
    with Path(outputs["summary_json"]).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(Path(outputs["report_md"]), summary)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "num_samples": len(rows),
                "mean_sparse_psnr_db": metrics["sparse_psnr_db"]["mean"],
                "mean_deq_mpi_psnr_db": metrics["deq_mpi_psnr_db"]["mean"],
                "mean_delta_sparse_minus_deq_mpi_db": metrics[
                    "delta_sparse_minus_deq_mpi_db"
                ]["mean"],
                "win_rate_vs_deq_mpi": summary["win_rates"]["delta_gt_0"],
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
