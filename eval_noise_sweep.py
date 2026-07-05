import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_deq_initialized_zeroshot import run_pnp_variant, save_refinement_panel
from make_2d_paper_comparison import (
    DEFAULT_CHECKPOINT,
    build_problem,
    run_deq,
    select_examples,
    setup_device,
)
from reconAlgos import psnr
from zeroshot_l1_pnp import build_denoiser


def parse_csv_floats(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the low-SNR sweep for DEQ-MPI and zero-shot refinement variants."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pSNRvals", default="15,10,5,0,-5")
    parser.add_argument("--examplePSNR", type=float, default=0.0)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--maxIter", type=int, default=25)
    parser.add_argument("--batchSizeDeq", type=int, default=256)
    parser.add_argument("--batchSizePnp", type=int, default=64)
    parser.add_argument("--testLimit", type=int, default=0)

    parser.add_argument("--zeroInitIter", type=int, default=7)
    parser.add_argument("--deqInitIter", type=int, default=1)
    parser.add_argument("--deqInitL1Iter", type=int, default=1)
    parser.add_argument("--l1Alpha", type=float, default=500.0)
    parser.add_argument("--mu0", type=float, default=5e5)
    parser.add_argument("--directSolve", action="store_true", default=True)
    parser.add_argument("--noDirectSolve", dest="directSolve", action="store_false")

    parser.add_argument("--denoiser", choices=["gaussian", "drunet"], default="drunet")
    parser.add_argument("--gaussianKernel", type=int, default=5)
    parser.add_argument("--gaussianSigma", type=float, default=1.0)
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")

    parser.add_argument("--numExamples", type=int, default=5)
    parser.add_argument(
        "--selection",
        choices=["deq_quantiles", "gt_energy_filtered_deq_quantiles"],
        default="gt_energy_filtered_deq_quantiles",
    )
    parser.add_argument("--minGtEnergyQuantile", type=float, default=0.5)
    parser.add_argument("--sampleIndices", default="")
    parser.add_argument("--cmap", default="magma")

    parser.add_argument("--outDir", default="training/reproduction/noise_sweep")
    parser.add_argument("--outCsv", default="training/reproduction/noise_sweep/noise_sweep_summary.csv")
    parser.add_argument(
        "--curvePath",
        default="training/reproduction/paper_figures/noise_sweep_psnr_curve.png",
    )
    parser.add_argument(
        "--figurePath",
        default="training/reproduction/paper_figures/low_snr_examples.png",
    )
    return parser.parse_args()


def method_metrics(method, init, n_iter, alpha, x_rec, elapsed, ref_vals, problem):
    sample_psnr = psnr(ref_vals, x_rec)
    residual = torch.linalg.norm(problem["y_reduced"] - x_rec @ problem["A_reduced"].T, dim=1)
    nrmse = torch.linalg.norm(ref_vals - x_rec, dim=1) / torch.linalg.norm(ref_vals, dim=1).clamp_min(1e-12)
    nonzero_ratio = (x_rec.abs() > 1e-8).float().mean(dim=1)
    return {
        "method": method,
        "init": init,
        "n_iter": int(n_iter),
        "alpha": float(alpha),
        "mean_psnr_db": float(np.mean(sample_psnr)),
        "std_psnr_db": float(np.std(sample_psnr)),
        "min_psnr_db": float(np.min(sample_psnr)),
        "max_psnr_db": float(np.max(sample_psnr)),
        "mean_nrmse": float(nrmse.mean().detach().cpu()),
        "mean_residual_norm": float(residual.mean().detach().cpu()),
        "mean_nonzero_ratio": float(nonzero_ratio.mean().detach().cpu()),
        "elapsed_seconds": float(elapsed),
    }, sample_psnr


def write_summary_csv(path, rows):
    fieldnames = [
        "pSNRval",
        "measured_snr_no_inverse_crime_db",
        "measured_snr_inverse_crime_db",
        "method",
        "init",
        "n_iter",
        "alpha",
        "mean_psnr_db",
        "std_psnr_db",
        "min_psnr_db",
        "max_psnr_db",
        "mean_nrmse",
        "mean_residual_norm",
        "mean_nonzero_ratio",
        "elapsed_seconds",
        "json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_curve(rows, out_path):
    method_order = [
        "LS/SVD",
        "DEQ-MPI",
        "ZeroShot-PnP",
        "DEQ-init ZeroShot-PnP",
        "DEQ-init ZeroShot-l1-PnP",
    ]
    colors = {
        "LS/SVD": "#7a7a7a",
        "DEQ-MPI": "#1f77b4",
        "ZeroShot-PnP": "#2ca02c",
        "DEQ-init ZeroShot-PnP": "#d62728",
        "DEQ-init ZeroShot-l1-PnP": "#9467bd",
    }
    markers = {
        "LS/SVD": "o",
        "DEQ-MPI": "o",
        "ZeroShot-PnP": "s",
        "DEQ-init ZeroShot-PnP": "^",
        "DEQ-init ZeroShot-l1-PnP": "D",
    }
    linestyles = {
        "LS/SVD": "-",
        "DEQ-MPI": "-",
        "ZeroShot-PnP": "-",
        "DEQ-init ZeroShot-PnP": "--",
        "DEQ-init ZeroShot-l1-PnP": "-.",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    for method in method_order:
        series = sorted([row for row in rows if row["method"] == method], key=lambda row: float(row["pSNRval"]))
        if not series:
            continue
        x = [float(row["pSNRval"]) for row in series]
        y = [float(row["mean_psnr_db"]) for row in series]
        yerr = [float(row["std_psnr_db"]) for row in series]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=markers.get(method, "o"),
            linestyle=linestyles.get(method, "-"),
            linewidth=1.8,
            capsize=2.5,
            label=method,
            color=colors.get(method),
        )

    ax.set_xlabel("Input pSNR setting (dB)")
    ax.set_ylabel("Mean reconstruction PSNR (dB)")
    ax.set_title("Low-SNR robustness on 2D simulated MPI")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sample_metrics(path, sample_rows):
    fieldnames = ["pSNRval", "sample_index", "method", "init", "n_iter", "alpha", "psnr_db"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)


def run_one_snr(args, p_snr, denoiser):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.pSNRval = float(p_snr)
    problem = build_problem(args, torch.device(f"cuda:{args.gpu}"))
    ref_vals = problem["ref_vals"]

    outputs = {}
    sample_psnr_by_key = {}
    rows = []

    ls_svd = problem["lsqr"]
    metrics, sample_psnr = method_metrics("LS/SVD", "none", 0, 0.0, ls_svd, 0.0, ref_vals, problem)
    rows.append(metrics)
    outputs["ls_svd"] = ls_svd
    sample_psnr_by_key["ls_svd"] = sample_psnr

    deq_mpi, deq_elapsed = run_deq(args, problem, torch.device(f"cuda:{args.gpu}"))
    metrics, sample_psnr = method_metrics("DEQ-MPI", "learned", args.maxIter, 0.0, deq_mpi, deq_elapsed, ref_vals, problem)
    rows.append(metrics)
    outputs["deq_mpi"] = deq_mpi
    sample_psnr_by_key["deq_mpi"] = sample_psnr

    zero = run_pnp_variant(args, problem, denoiser, n_iter=args.zeroInitIter, alpha=0.0, init_name="zero", x0=None)
    metrics, sample_psnr = method_metrics(
        "ZeroShot-PnP",
        "zero",
        args.zeroInitIter,
        0.0,
        zero["x_rec"],
        zero["elapsed_seconds"],
        ref_vals,
        problem,
    )
    rows.append(metrics)
    outputs["zeroshot_pnp"] = zero["x_rec"]
    sample_psnr_by_key["zeroshot_pnp"] = sample_psnr

    deq_init = run_pnp_variant(args, problem, denoiser, n_iter=args.deqInitIter, alpha=0.0, init_name="DEQ-MPI", x0=deq_mpi)
    metrics, sample_psnr = method_metrics(
        "DEQ-init ZeroShot-PnP",
        "DEQ-MPI",
        args.deqInitIter,
        0.0,
        deq_init["x_rec"],
        deq_init["elapsed_seconds"],
        ref_vals,
        problem,
    )
    rows.append(metrics)
    outputs["deq_init_pnp"] = deq_init["x_rec"]
    sample_psnr_by_key["deq_init_pnp"] = sample_psnr

    deq_init_l1 = run_pnp_variant(
        args,
        problem,
        denoiser,
        n_iter=args.deqInitL1Iter,
        alpha=args.l1Alpha,
        init_name="DEQ-MPI",
        x0=deq_mpi,
    )
    metrics, sample_psnr = method_metrics(
        "DEQ-init ZeroShot-l1-PnP",
        "DEQ-MPI",
        args.deqInitL1Iter,
        args.l1Alpha,
        deq_init_l1["x_rec"],
        deq_init_l1["elapsed_seconds"],
        ref_vals,
        problem,
    )
    rows.append(metrics)
    outputs["deq_init_l1_pnp"] = deq_init_l1["x_rec"]
    sample_psnr_by_key["deq_init_l1_pnp"] = sample_psnr

    for row in rows:
        row["pSNRval"] = float(p_snr)
        row["measured_snr_no_inverse_crime_db"] = problem["measured_snr_no_inverse_crime_db"]
        row["measured_snr_inverse_crime_db"] = problem["measured_snr_inverse_crime_db"]

    return problem, ref_vals, outputs, sample_psnr_by_key, rows


def main():
    args = parse_args()
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.outCsv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_path = Path(args.curvePath)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figurePath)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    device = setup_device(args.gpu)
    denoiser_args = SimpleNamespace(
        denoiser=args.denoiser,
        gaussianKernel=args.gaussianKernel,
        gaussianSigma=args.gaussianSigma,
        dpirRoot=args.dpirRoot,
        drunetWeights=args.drunetWeights,
    )
    denoiser = build_denoiser(denoiser_args, device)

    all_rows = []
    all_sample_rows = []
    example_payload = None
    start_time = time.time()
    p_snr_values = parse_csv_floats(args.pSNRvals)

    for p_snr in p_snr_values:
        problem, ref_vals, outputs, sample_psnr_by_key, rows = run_one_snr(args, p_snr, denoiser)
        p_tag = f"{p_snr:g}".replace("-", "m").replace(".", "p")
        json_path = out_dir / f"noise_sweep_pSNR{p_tag}.json"
        payload = {
            "seed": args.seed,
            "pSNRval": float(p_snr),
            "num_test_images": problem["num_test_images"],
            "denoiser": args.denoiser,
            "nbOfSingulars": args.nbOfSingulars,
            "methods": rows,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        for row in rows:
            row["json"] = str(json_path)
        all_rows.extend(rows)

        key_to_row = {
            "ls_svd": rows[0],
            "deq_mpi": rows[1],
            "zeroshot_pnp": rows[2],
            "deq_init_pnp": rows[3],
            "deq_init_l1_pnp": rows[4],
        }
        for key, values in sample_psnr_by_key.items():
            row = key_to_row[key]
            for idx, value in enumerate(values):
                all_sample_rows.append(
                    {
                        "pSNRval": float(p_snr),
                        "sample_index": idx,
                        "method": row["method"],
                        "init": row["init"],
                        "n_iter": row["n_iter"],
                        "alpha": row["alpha"],
                        "psnr_db": float(value),
                    }
                )

        if example_payload is None or abs(float(p_snr) - args.examplePSNR) < abs(float(example_payload["p_snr"]) - args.examplePSNR):
            example_payload = {
                "p_snr": float(p_snr),
                "problem": problem,
                "ref_vals": ref_vals,
                "outputs": outputs,
                "sample_psnr_by_key": sample_psnr_by_key,
            }

    all_rows = sorted(all_rows, key=lambda row: (float(row["pSNRval"]), row["method"]))
    write_summary_csv(out_csv, all_rows)
    write_sample_metrics(out_dir / "noise_sweep_all_sample_metrics.csv", all_sample_rows)
    save_curve(all_rows, curve_path)

    if example_payload is not None:
        problem = example_payload["problem"]
        ref_vals = example_payload["ref_vals"]
        selected_indices = select_examples(args, example_payload["sample_psnr_by_key"]["deq_mpi"], ref_vals)
        panel_outputs = {
            "images": {
                "ground_truth": ref_vals,
                "deq_mpi": example_payload["outputs"]["deq_mpi"],
                "zeroshot_pnp": example_payload["outputs"]["zeroshot_pnp"],
                "deq_init_pnp": example_payload["outputs"]["deq_init_pnp"],
                "deq_init_l1_pnp": example_payload["outputs"]["deq_init_l1_pnp"],
            },
            "order": [
                ("Ground truth", "ground_truth"),
                ("DEQ-MPI", "deq_mpi"),
                (f"ZeroShot {args.zeroInitIter} it", "zeroshot_pnp"),
                (f"DEQ-init PnP {args.deqInitIter} it", "deq_init_pnp"),
                (f"DEQ-init l1 {args.deqInitL1Iter} it", "deq_init_l1_pnp"),
            ],
        }
        save_refinement_panel(
            args,
            problem,
            panel_outputs,
            example_payload["sample_psnr_by_key"],
            selected_indices,
            figure_path,
            title=f"Low-SNR examples at pSNR={example_payload['p_snr']:g} dB",
        )
    else:
        selected_indices = []

    manifest = {
        "seed": args.seed,
        "pSNRvals": p_snr_values,
        "examplePSNR": None if example_payload is None else example_payload["p_snr"],
        "selected_indices": selected_indices,
        "denoiser": args.denoiser,
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "summary_csv": str(out_csv),
            "all_sample_metrics_csv": str(out_dir / "noise_sweep_all_sample_metrics.csv"),
            "curve_png": str(curve_path),
            "low_snr_examples_png": str(figure_path),
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
