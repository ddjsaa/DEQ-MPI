"""Evaluate a trained Sparse-PnP-ADMM-DEQ checkpoint.

Example:
  python eval_sparse_pnp_admm_deq_checkpoint.py \
    --runDir training/reproduction/sparse_pnp_admm_deq/finite_rdn_smoke \
    --split val --limit 128

The evaluator reconstructs the model from the run's args.json, loads the
checkpoint, and writes aggregate JSON plus per-sample CSV metrics.
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
from torch.utils.data import DataLoader

from sparse_pnp_admm_deq import SparsePnPADMMDEQ, SparsePnPADMMFixedPoint
from train_sparse_pnp_admm_deq import (
    build_system,
    load_tensor_dataset,
    make_denoiser,
    prepare_batch,
    setup_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Sparse-PnP-ADMM-DEQ checkpoint")
    parser.add_argument("--runDir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--argsJson", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--dataPath", default="")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--outDir", default="")
    parser.add_argument("--maxIterOverride", type=int, default=0)
    parser.add_argument("--solverOverride", choices=["", "finite", "deq"], default="")
    parser.add_argument("--deqSolverOverride", choices=["", "fixed", "regular", "anderson"], default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_run_args(args):
    run_dir = Path(args.runDir)
    args_json = Path(args.argsJson) if args.argsJson else run_dir / "args.json"
    with args_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["gpu"] = args.gpu
    payload["seed"] = args.seed
    if args.maxIterOverride > 0:
        payload["maxIter"] = args.maxIterOverride
    if args.solverOverride:
        payload["solver"] = args.solverOverride
    if args.deqSolverOverride:
        payload["deqSolver"] = args.deqSolverOverride
    return SimpleNamespace(**payload), args_json


def data_path_for_split(run_args, split: str, override: str) -> str:
    if override:
        return override
    if split == "train":
        return run_args.trainPath
    if split == "val":
        return run_args.valPath
    return "datasets/testPatches.h5"


def checkpoint_path(args):
    run_dir = Path(args.runDir)
    if args.checkpoint:
        return Path(args.checkpoint)
    best = run_dir / "best.pth"
    if best.exists():
        return best
    finals = sorted(run_dir.glob("epoch*END.pth"))
    if not finals:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return finals[-1]


def build_model(run_args, system, device):
    denoiser = make_denoiser(run_args, device)
    fixed_point = SparsePnPADMMFixedPoint(
        A=system["a_reduced"],
        image_shape=system["img_size"],
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
        solver_kwargs={"tol": 1e-4, "beta": 1.0} if run_args.deqSolver in ("regular", "anderson") else None,
    ).to(device)
    return model, fixed_point


def per_sample_metrics(target, pred, y_reduced, A_reduced, start_index):
    n_pix = target.shape[1]
    err = torch.linalg.norm(target - pred, dim=1).clamp_min(1e-12)
    target_norm = torch.linalg.norm(target, dim=1).clamp_min(1e-12)
    peak = target.max().clamp_min(1e-12)
    psnr = 10 * torch.log10(n_pix * peak.square() / err.square())
    nrmse = err / target_norm
    residual = torch.linalg.norm(y_reduced - pred @ A_reduced.T, dim=1)
    nonzero = (pred.abs() > 1e-8).float().mean(dim=1)
    rows = []
    for local_idx in range(target.shape[0]):
        rows.append(
            {
                "sample_index": start_index + local_idx,
                "psnr_db": float(psnr[local_idx].detach().cpu()),
                "nrmse": float(nrmse[local_idx].detach().cpu()),
                "data_residual_norm": float(residual[local_idx].detach().cpu()),
                "nonzero_ratio": float(nonzero[local_idx].detach().cpu()),
                "min_value": float(pred[local_idx].min().detach().cpu()),
                "max_value": float(pred[local_idx].max().detach().cpu()),
            }
        )
    return rows


def summarize_rows(rows, elapsed):
    fields = ["psnr_db", "nrmse", "data_residual_norm", "nonzero_ratio", "min_value", "max_value"]
    summary = {"num_samples": len(rows), "elapsed_seconds": float(elapsed)}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        summary[f"mean_{field}"] = float(values.mean())
        summary[f"std_{field}"] = float(values.std())
        summary[f"min_{field}"] = float(values.min())
        summary[f"max_{field}"] = float(values.max())
    return summary


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = setup_device(args.gpu)
    run_args, args_json = load_run_args(args)
    ckpt = checkpoint_path(args)

    system = build_system(device, run_args.nbOfSingulars)
    data_path = data_path_for_split(run_args, args.split, args.dataPath)
    dataset = load_tensor_dataset(data_path, args.limit)
    loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False)
    model, fixed_point = build_model(run_args, system, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    rows = []
    first_probe = None
    start = time.time()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            y_reduced, x0, target = prepare_batch(batch, system, run_args, device, train=False)
            if first_probe is None:
                if run_args.solver == "finite":
                    pred, state, history = model(y_reduced, x0=x0, return_state=True, return_history=True)
                    first_probe = history
                else:
                    pred, state = model(y_reduced, x0=x0, return_state=True)
                    residuals = fixed_point.residuals(state, y_reduced)
                    first_probe = [
                        {
                            "iter": "deq_final",
                            "fixed_point_abs_mean": float(residuals["fixed_point_abs"].mean().detach().cpu()),
                            "fixed_point_rel_mean": float(residuals["fixed_point_rel"].mean().detach().cpu()),
                            "primal_z_mean": float(residuals["primal_z"].mean().detach().cpu()),
                            "primal_s_mean": float(residuals["primal_s"].mean().detach().cpu()),
                            "data_residual_mean": float(residuals["data_residual"].mean().detach().cpu()),
                        }
                    ]
            else:
                pred = model(y_reduced, x0=x0)
            rows.extend(per_sample_metrics(target, pred, y_reduced, system["a_reduced"], seen))
            seen += target.shape[0]
    elapsed = time.time() - start

    finite_output = all(np.isfinite(row["psnr_db"]) and np.isfinite(row["nrmse"]) for row in rows)
    nonnegative_ok = bool(run_args.allowNegative or min(row["min_value"] for row in rows) >= -1e-7)
    acceptance = {
        "finite_output": finite_output,
        "nonnegative_ok": nonnegative_ok,
        "num_samples_ok": len(rows) == len(dataset),
        "passed": finite_output and nonnegative_ok and len(rows) == len(dataset),
    }

    run_dir = Path(args.runDir)
    out_dir = Path(args.outDir) if args.outDir else run_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{args.split}_{ckpt.stem}_n{len(rows)}"
    sample_csv = out_dir / f"{prefix}_sample_metrics.csv"
    summary_json = out_dir / f"{prefix}_summary.json"
    write_csv(sample_csv, rows)

    summary = {
        "method": "Sparse-PnP-ADMM-DEQ",
        "runDir": str(run_dir),
        "argsJson": str(args_json),
        "checkpoint": str(ckpt),
        "split": args.split,
        "dataPath": data_path,
        "limit": args.limit,
        "seed": args.seed,
        "solver": run_args.solver,
        "deqSolver": run_args.deqSolver if run_args.solver == "deq" else None,
        "maxIter": run_args.maxIter,
        "denoiser": run_args.denoiser,
        "rho": run_args.rho,
        "eta": run_args.eta,
        "l1Lambda": run_args.l1Lambda,
        "acceptance": acceptance,
        "metrics": summarize_rows(rows, elapsed),
        "fixed_point_probe": first_probe,
        "outputs": {
            "summary_json": str(summary_json),
            "sample_metrics_csv": str(sample_csv),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"acceptance": acceptance, "metrics": summary["metrics"]}, indent=2))
    if args.strict and not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
