"""Smoke-test and evaluate Sparse-PnP-ADMM-DEQ on the 2D MPI benchmark.

Example:
  python eval_sparse_pnp_admm_deq.py --testLimit 32 --maxIter 12

Outputs:
  - summary JSON with configuration, acceptance checks, and aggregate metrics
  - per-sample CSV with PSNR values for LS/SVD, proposed method, and optional DEQ-MPI
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
import torch.nn as nn

from make_2d_paper_comparison import DEFAULT_CHECKPOINT, build_problem, run_deq, setup_device
from reconAlgos import psnr
from sparse_pnp_admm_deq import SparsePnPADMMDEQ, SparsePnPADMMFixedPoint
from zeroshot_l1_pnp import GaussianDenoiser, build_denoiser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--testLimit", type=int, default=32)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--maxIter", type=int, default=12)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--l1Lambda", type=float, default=0.0005)
    parser.add_argument("--allowNegative", action="store_true")
    parser.add_argument("--normalizeDenoiserInput", action="store_true")
    parser.add_argument("--denoiserSigma", type=float, default=None)
    parser.add_argument("--denoiser", choices=["identity", "gaussian", "drunet"], default="gaussian")
    parser.add_argument("--gaussianKernel", type=int, default=3)
    parser.add_argument("--gaussianSigma", type=float, default=0.65)
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")
    parser.add_argument("--batchSizeDeq", type=int, default=256)
    parser.add_argument("--deqMaxIter", type=int, default=25)
    parser.add_argument("--deqCheckpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skipDeq", action="store_true")
    parser.add_argument("--outDir", default="training/reproduction/sparse_pnp_admm_deq")
    parser.add_argument("--prefix", default="smoke")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--residualBound", type=float, default=1e6)
    return parser.parse_args()


def make_denoiser(args, device):
    if args.denoiser == "identity":
        return nn.Identity().to(device).eval()
    if args.denoiser == "gaussian":
        return GaussianDenoiser(kernel_size=args.gaussianKernel, sigma=args.gaussianSigma).to(device).eval()
    return build_denoiser(args, device)


def summarize_psnr(name, values, elapsed=0.0):
    values = np.asarray(values, dtype=np.float64)
    return {
        "method": name,
        "mean_psnr_db": float(values.mean()),
        "std_psnr_db": float(values.std()),
        "min_psnr_db": float(values.min()),
        "max_psnr_db": float(values.max()),
        "elapsed_seconds": float(elapsed),
    }


def tensor_metrics(x_rec, ref_vals, y_reduced, A_reduced):
    nrmse = torch.linalg.norm(ref_vals - x_rec, dim=1) / torch.linalg.norm(ref_vals, dim=1).clamp_min(1e-12)
    residual = torch.linalg.norm(y_reduced - x_rec @ A_reduced.T, dim=1)
    nonzero_ratio = (x_rec.abs() > 1e-8).float().mean(dim=1)
    return {
        "mean_nrmse": float(nrmse.mean().detach().cpu()),
        "mean_data_residual_norm": float(residual.mean().detach().cpu()),
        "mean_nonzero_ratio": float(nonzero_ratio.mean().detach().cpu()),
        "min_value": float(x_rec.min().detach().cpu()),
        "max_value": float(x_rec.max().detach().cpu()),
    }


def write_sample_csv(path, rows):
    fieldnames = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = setup_device(args.gpu)

    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = build_problem(args, device)
    ref_vals = problem["ref_vals"]
    ls_svd = problem["lsqr"]
    denoiser = make_denoiser(args, device)

    fp = SparsePnPADMMFixedPoint(
        A=problem["A_reduced"],
        image_shape=problem["img_size"],
        denoiser=denoiser,
        rho=args.rho,
        eta=args.eta,
        l1_lambda=args.l1Lambda,
        non_negative=not args.allowNegative,
        normalize_denoiser_input=args.normalizeDenoiserInput,
        denoiser_sigma=args.denoiserSigma,
    ).to(device)
    model = SparsePnPADMMDEQ(fp, max_iter=args.maxIter, solver="finite").to(device).eval()

    start_time = time.time()
    with torch.no_grad():
        x_rec, state, history = model(
            problem["y_reduced"],
            x0=ls_svd,
            return_state=True,
            return_history=True,
        )
    elapsed = time.time() - start_time

    ls_psnr = psnr(ref_vals, ls_svd)
    proposed_psnr = psnr(ref_vals, x_rec)
    deq_psnr = None
    deq_elapsed = None
    deq_rec = None
    deq_checkpoint = Path(args.deqCheckpoint)
    if not args.skipDeq and deq_checkpoint.exists():
        deq_args = SimpleNamespace(
            checkpoint=str(deq_checkpoint),
            maxIter=args.deqMaxIter,
            batchSizeDeq=args.batchSizeDeq,
        )
        deq_rec, deq_elapsed = run_deq(deq_args, problem, device)
        deq_psnr = psnr(ref_vals, deq_rec)

    finite_output = bool(torch.isfinite(x_rec).all().detach().cpu())
    shape_ok = tuple(x_rec.shape) == tuple(ref_vals.shape)
    nonnegative_ok = bool(args.allowNegative or x_rec.min().detach().cpu() >= -1e-7)
    history_finite = all(np.isfinite(row["fixed_point_rel_mean"]) for row in history)
    residual_bounded = history_finite and max(row["fixed_point_rel_mean"] for row in history) < args.residualBound
    psnr_above_ls = float(np.mean(proposed_psnr)) > float(np.mean(ls_psnr))
    acceptance = {
        "finite_output": finite_output,
        "shape_ok": shape_ok,
        "nonnegative_ok": nonnegative_ok,
        "history_finite": history_finite,
        "fixed_point_residual_bounded": residual_bounded,
        "mean_psnr_above_ls_svd": psnr_above_ls,
        "passed": all([finite_output, shape_ok, nonnegative_ok, history_finite, residual_bounded, psnr_above_ls]),
    }

    sample_rows = []
    for idx, value in enumerate(proposed_psnr):
        row = {
            "sample_index": idx,
            "ls_svd_psnr_db": float(ls_psnr[idx]),
            "sparse_pnp_admm_deq_psnr_db": float(value),
        }
        if deq_psnr is not None:
            row["deq_mpi_psnr_db"] = float(deq_psnr[idx])
        sample_rows.append(row)

    sample_csv = out_dir / f"{args.prefix}_sample_metrics.csv"
    summary_json = out_dir / f"{args.prefix}_summary.json"
    write_sample_csv(sample_csv, sample_rows)

    summary = {
        "seed": args.seed,
        "num_test_images": problem["num_test_images"],
        "pSNRval": args.pSNRval,
        "measured_snr_no_inverse_crime_db": problem["measured_snr_no_inverse_crime_db"],
        "measured_snr_inverse_crime_db": problem["measured_snr_inverse_crime_db"],
        "nbOfSingulars": args.nbOfSingulars,
        "method": "Sparse-PnP-ADMM-DEQ",
        "config": {
            "maxIter": args.maxIter,
            "rho": args.rho,
            "eta": args.eta,
            "l1Lambda": args.l1Lambda,
            "non_negative": not args.allowNegative,
            "denoiser": args.denoiser,
            "gaussianKernel": args.gaussianKernel,
            "gaussianSigma": args.gaussianSigma,
            "normalizeDenoiserInput": args.normalizeDenoiserInput,
            "denoiserSigma": args.denoiserSigma,
        },
        "acceptance": acceptance,
        "metrics": {
            "ls_svd": summarize_psnr("LS/SVD", ls_psnr),
            "sparse_pnp_admm_deq": {
                **summarize_psnr("Sparse-PnP-ADMM-DEQ", proposed_psnr, elapsed),
                **tensor_metrics(x_rec, ref_vals, problem["y_reduced"], problem["A_reduced"]),
            },
        },
        "fixed_point_history": history,
        "outputs": {
            "summary_json": str(summary_json),
            "sample_metrics_csv": str(sample_csv),
        },
    }
    if deq_psnr is not None:
        summary["metrics"]["deq_mpi"] = summarize_psnr("DEQ-MPI", deq_psnr, deq_elapsed)
        summary["deqCheckpoint"] = str(deq_checkpoint)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["acceptance"], indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    if args.strict and not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
