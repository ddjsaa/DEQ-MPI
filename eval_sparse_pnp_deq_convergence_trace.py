"""Record paired Sparse-PnP-DEQ convergence traces.

Example:
  python eval_sparse_pnp_deq_convergence_trace.py --mode canonical

The script reuses the paired evaluation problem construction, then records
Anderson solver residual traces and final per-sample fixed-point diagnostics for
the quality and fast Sparse-PnP-DEQ settings.
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

from diagnose_sparse_pnp_admm_deq_gap import stats
from eval_paired_sparse_pnp_deqmpi import (
    DEFAULT_SPARSE_RUN,
    build_problem,
    build_sparse_model,
    load_run_args,
    load_sparse_weights,
    nrmse_values,
    setup_device,
    sparse_checkpoint_path,
)


def psnr_values_with_peak(target, pred, peak):
    n_pix = target.shape[1]
    err = torch.linalg.norm(target - pred, dim=1).clamp_min(1e-12)
    return 10 * torch.log10(n_pix * peak.square() / err.square())


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse-PnP-DEQ convergence trace evaluator")
    parser.add_argument("--sparseRunDir", default=str(DEFAULT_SPARSE_RUN))
    parser.add_argument("--sparseCheckpoint", default="")
    parser.add_argument("--mode", choices=["direct", "canonical"], default="canonical")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--testPath", default="datasets/testPatches.h5")
    parser.add_argument("--testLimit", type=int, default=0)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--nbOfSingulars", type=int, default=0)
    parser.add_argument("--pSNRval", type=float, default=-1.0)
    parser.add_argument("--rescaleMin", type=float, default=-1.0)
    parser.add_argument("--rescaleMax", type=float, default=-1.0)
    parser.add_argument("--outDir", default=str(DEFAULT_SPARSE_RUN / "convergence_traces"))
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def problem_args(args):
    return SimpleNamespace(
        mode=args.mode,
        testPath=args.testPath,
        testLimit=args.testLimit,
        nbOfSingulars=args.nbOfSingulars,
        pSNRval=args.pSNRval,
        rescaleMin=args.rescaleMin,
        rescaleMax=args.rescaleMax,
    )


def settings():
    return [
        {
            "name": "quality_it12_eta0p10",
            "solver": "deq",
            "deqSolver": "anderson",
            "maxIter": 12,
            "rho": 1.0,
            "eta": 0.10,
            "l1Lambda": 0.0005,
        },
        {
            "name": "fast_it10_eta0p001",
            "solver": "deq",
            "deqSolver": "anderson",
            "maxIter": 10,
            "rho": 1.0,
            "eta": 0.001,
            "l1Lambda": 0.0005,
        },
    ]


def apply_setting(base_args, setting):
    run_args = SimpleNamespace(**vars(base_args))
    for key in ["solver", "deqSolver", "maxIter", "rho", "eta", "l1Lambda"]:
        setattr(run_args, key, setting[key])
    return run_args


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_trace(histories):
    max_len = max((len(hist) for hist in histories), default=0)
    rows = []
    for idx in range(max_len):
        vals = np.asarray([hist[idx] for hist in histories if len(hist) > idx], dtype=np.float64)
        if vals.size == 0:
            continue
        rows.append(
            {
                "solver_iter": idx + 2,
                "num_batches": int(vals.size),
                "mean_solver_residual": float(vals.mean()),
                "std_solver_residual": float(vals.std()),
                "min_solver_residual": float(vals.min()),
                "max_solver_residual": float(vals.max()),
                "median_solver_residual": float(np.median(vals)),
                "q10_solver_residual": float(np.quantile(vals, 0.10)),
                "q90_solver_residual": float(np.quantile(vals, 0.90)),
            }
        )
    return rows


def as_float_list(tensor):
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def run_setting(setting, base_run_args, problem, checkpoint, device, batch_size):
    run_args = apply_setting(base_run_args, setting)
    model, fixed_point = build_sparse_model(run_args, problem, device)
    sparse_load = load_sparse_weights(model, checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    target = problem["target"]
    y_reduced = problem["y_reduced"]
    x0 = problem["x0"]
    global_peak = target.max().clamp_min(1e-12)

    sample_rows = []
    histories = []
    seen = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, y_reduced.shape[0], batch_size):
            end = min(start + batch_size, y_reduced.shape[0])
            pred, state = model(
                y_reduced[start:end],
                x0=x0[start:end],
                return_state=True,
            )
            if model.deq_layer is not None and isinstance(model.deq_layer.forward_res, list):
                histories.append([float(value) for value in model.deq_layer.forward_res])
            diagnostics = fixed_point.residuals(state, y_reduced[start:end])
            psnr = psnr_values_with_peak(target[start:end], pred, global_peak)
            nrmse = nrmse_values(target[start:end], pred)
            fields = {
                "psnr_db": as_float_list(psnr),
                "nrmse": as_float_list(nrmse),
                "fixed_point_abs": as_float_list(diagnostics["fixed_point_abs"]),
                "fixed_point_rel": as_float_list(diagnostics["fixed_point_rel"]),
                "primal_z": as_float_list(diagnostics["primal_z"]),
                "primal_s": as_float_list(diagnostics["primal_s"]),
                "data_residual": as_float_list(diagnostics["data_residual"]),
                "pred_min_value": as_float_list(pred.min(dim=1).values),
                "pred_max_value": as_float_list(pred.max(dim=1).values),
            }
            for local_idx in range(end - start):
                row = {"setting": setting["name"], "sample_index": seen + local_idx}
                for key, values in fields.items():
                    row[key] = values[local_idx]
                sample_rows.append(row)
            seen += end - start
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.time() - start_time

    trace_rows = summarize_trace(histories)
    for row in trace_rows:
        row["setting"] = setting["name"]

    metric_fields = [
        "psnr_db",
        "nrmse",
        "fixed_point_abs",
        "fixed_point_rel",
        "primal_z",
        "primal_s",
        "data_residual",
        "pred_min_value",
        "pred_max_value",
    ]
    metrics = {}
    for field in metric_fields:
        metrics[field] = stats(np.asarray([row[field] for row in sample_rows], dtype=np.float64))

    trace_residuals = np.asarray(
        [row["mean_solver_residual"] for row in trace_rows], dtype=np.float64
    )
    trace_summary = {
        "num_batches": len(histories),
        "num_solver_trace_points": len(trace_rows),
        "first_mean_solver_residual": float(trace_residuals[0]) if trace_residuals.size else None,
        "final_mean_solver_residual": float(trace_residuals[-1]) if trace_residuals.size else None,
        "min_mean_solver_residual": float(trace_residuals.min()) if trace_residuals.size else None,
    }

    acceptance = {
        "finite_output": bool(np.isfinite([row["psnr_db"] for row in sample_rows]).all()),
        "nonnegative_ok": bool(min(row["pred_min_value"] for row in sample_rows) >= -1e-7),
        "num_samples_ok": len(sample_rows) == int(target.shape[0]),
    }
    acceptance["passed"] = all(acceptance.values())

    summary = {
        "setting": setting,
        "load": sparse_load,
        "num_samples": len(sample_rows),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "trace": trace_summary,
        "acceptance": acceptance,
    }
    return sample_rows, trace_rows, summary


def write_report(path: Path, summary):
    lines = [
        "# Sparse-PnP-DEQ Convergence Trace",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Samples: `{summary['num_samples']}`",
        "",
        "## Settings",
    ]
    for item in summary["settings"]:
        metrics = item["metrics"]
        trace = item["trace"]
        lines.append(
            "- "
            f"`{item['setting']['name']}`: PSNR `{metrics['psnr_db']['mean']:.4f} dB`, "
            f"fixed-point rel `{metrics['fixed_point_rel']['mean']:.5f}`, "
            f"final solver residual `{trace['final_mean_solver_residual']:.6f}`, "
            f"elapsed `{item['elapsed_seconds']:.3f}s`."
        )
    lines.extend(["", "## Outputs"])
    for name, path_value in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path_value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.sparseRunDir)
    checkpoint = sparse_checkpoint_path(args, run_dir)
    base_run_args = load_run_args(run_dir)

    device = setup_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    problem = build_problem(problem_args(args), base_run_args, device)

    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"convergence_{args.mode}_seed{args.seed}"

    all_sample_rows = []
    all_trace_rows = []
    setting_summaries = []
    for setting in settings():
        sample_rows, trace_rows, setting_summary = run_setting(
            setting,
            base_run_args,
            problem,
            checkpoint,
            device,
            args.batchSize,
        )
        all_sample_rows.extend(sample_rows)
        all_trace_rows.extend(trace_rows)
        setting_summaries.append(setting_summary)

    sample_csv = out_dir / f"{prefix}_sample_final_residuals.csv"
    trace_csv = out_dir / f"{prefix}_solver_trace.csv"
    summary_json = out_dir / f"{prefix}_summary.json"
    report_md = out_dir / f"{prefix}_report.md"
    write_csv(sample_csv, all_sample_rows)
    write_csv(trace_csv, all_trace_rows)

    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "num_samples": int(problem["target"].shape[0]),
        "problem": {
            "testPath": args.testPath,
            "pSNRval": problem["pSNRval"],
            "measured_snr_db": problem["measured_snr_db"],
            "rescaleMin": problem["rescaleMin"],
            "rescaleMax": problem["rescaleMax"],
            "nbOfSingulars": problem["nbOfSingulars"],
        },
        "settings": setting_summaries,
        "outputs": {
            "sample_final_residuals_csv": str(sample_csv),
            "solver_trace_csv": str(trace_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(report_md, summary)

    print(
        json.dumps(
            {
                "num_samples": summary["num_samples"],
                "settings": [
                    {
                        "name": item["setting"]["name"],
                        "mean_psnr_db": item["metrics"]["psnr_db"]["mean"],
                        "mean_fixed_point_rel": item["metrics"]["fixed_point_rel"]["mean"],
                        "final_mean_solver_residual": item["trace"][
                            "final_mean_solver_residual"
                        ],
                        "elapsed_seconds": item["elapsed_seconds"],
                    }
                    for item in setting_summaries
                ],
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
