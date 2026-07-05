import argparse
import csv
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_deq_initialized_zeroshot import run_pnp_variant
from eval_noise_sweep import method_metrics
from make_2d_paper_comparison import DEFAULT_CHECKPOINT, build_problem, run_deq, setup_device
from zeroshot_l1_pnp import build_denoiser


DEFAULT_CHECKPOINT_DIR = str(Path(DEFAULT_CHECKPOINT).parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate training-budget dependence using saved DEQ-MPI epoch checkpoints, "
            "with zero-training ZeroShot-PnP and fixed DEQ-initialized refinement."
        )
    )
    parser.add_argument("--checkpointDir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", default="10,20,50,100,150,200END")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--maxIter", type=int, default=25)
    parser.add_argument("--batchSizeDeq", type=int, default=256)
    parser.add_argument("--batchSizePnp", type=int, default=64)
    parser.add_argument("--testLimit", type=int, default=0)

    parser.add_argument("--zeroInitIter", type=int, default=7)
    parser.add_argument("--deqInitIter", type=int, default=1)
    parser.add_argument("--mu0", type=float, default=5e5)
    parser.add_argument("--directSolve", action="store_true", default=True)
    parser.add_argument("--noDirectSolve", dest="directSolve", action="store_false")

    parser.add_argument("--denoiser", choices=["gaussian", "drunet"], default="drunet")
    parser.add_argument("--gaussianKernel", type=int, default=5)
    parser.add_argument("--gaussianSigma", type=float, default=1.0)
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")

    parser.add_argument("--outDir", default="training/reproduction/few_training")
    parser.add_argument("--outCsv", default="training/reproduction/few_training/few_training_summary.csv")
    parser.add_argument(
        "--figurePath",
        default="training/reproduction/paper_figures/few_training_curve.png",
    )
    return parser.parse_args()


def epoch_label_to_path(checkpoint_dir, label):
    label = str(label).strip()
    path = Path(checkpoint_dir) / f"epoch{label}.pth"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found for epoch label {label}: {path}")
    return path


def epoch_label_to_x(label):
    match = re.search(r"\d+", str(label))
    if not match:
        raise ValueError(f"could not parse numeric epoch from label: {label}")
    return int(match.group(0))


def parse_epoch_labels(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def write_summary_csv(path, rows):
    fieldnames = [
        "training_regime",
        "epoch_label",
        "epoch",
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
        "runtime_per_image_seconds",
        "checkpoint",
        "json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_curve(rows, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    style = {
        "ZeroShot-PnP": {"color": "#2ca02c", "marker": "s", "linestyle": "none"},
        "DEQ-MPI": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "DEQ-init ZeroShot-PnP": {"color": "#d62728", "marker": "^", "linestyle": "--"},
    }
    for method in ["ZeroShot-PnP", "DEQ-MPI", "DEQ-init ZeroShot-PnP"]:
        series = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["epoch"]))
        if not series:
            continue
        x = [int(row["epoch"]) for row in series]
        y = [float(row["mean_psnr_db"]) for row in series]
        yerr = [float(row["std_psnr_db"]) for row in series]
        ax.errorbar(x, y, yerr=yerr, capsize=2.5, linewidth=1.8, label=method, **style[method])

    ax.set_xlabel("Supervised DEQ-MPI training epoch")
    ax.set_ylabel("Mean reconstruction PSNR (dB)")
    ax.set_title("Training-budget dependence on 2D simulated MPI")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_common_fields(row, args, problem, checkpoint="", epoch_label="0", training_regime="zero paired training"):
    row = dict(row)
    row["training_regime"] = training_regime
    row["epoch_label"] = str(epoch_label)
    row["epoch"] = epoch_label_to_x(epoch_label)
    row["runtime_per_image_seconds"] = row["elapsed_seconds"] / max(1, problem["num_test_images"])
    row["checkpoint"] = str(checkpoint)
    row["json"] = ""
    return row


def main():
    args = parse_args()
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.outCsv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figurePath)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = setup_device(args.gpu)
    problem = build_problem(args, device)
    ref_vals = problem["ref_vals"]

    denoiser_args = SimpleNamespace(
        denoiser=args.denoiser,
        gaussianKernel=args.gaussianKernel,
        gaussianSigma=args.gaussianSigma,
        dpirRoot=args.dpirRoot,
        drunetWeights=args.drunetWeights,
    )
    denoiser = build_denoiser(denoiser_args, device)

    rows = []
    sample_rows = []
    start_time = time.time()

    zero = run_pnp_variant(args, problem, denoiser, n_iter=args.zeroInitIter, alpha=0.0, init_name="zero", x0=None)
    zero_metrics, zero_sample_psnr = method_metrics(
        "ZeroShot-PnP",
        "zero",
        args.zeroInitIter,
        0.0,
        zero["x_rec"],
        zero["elapsed_seconds"],
        ref_vals,
        problem,
    )
    zero_row = add_common_fields(zero_metrics, args, problem, epoch_label="0", training_regime="zero paired training")
    rows.append(zero_row)
    for idx, value in enumerate(zero_sample_psnr):
        sample_rows.append(
            {
                "epoch_label": "0",
                "epoch": 0,
                "method": "ZeroShot-PnP",
                "sample_index": idx,
                "psnr_db": float(value),
            }
        )

    for label in parse_epoch_labels(args.epochs):
        checkpoint = epoch_label_to_path(args.checkpointDir, label)
        args.checkpoint = str(checkpoint)
        deq_mpi, deq_elapsed = run_deq(args, problem, device)

        deq_metrics, deq_sample_psnr = method_metrics(
            "DEQ-MPI",
            "learned",
            args.maxIter,
            0.0,
            deq_mpi,
            deq_elapsed,
            ref_vals,
            problem,
        )
        deq_row = add_common_fields(
            deq_metrics,
            args,
            problem,
            checkpoint=checkpoint,
            epoch_label=label,
            training_regime="supervised checkpoint",
        )
        rows.append(deq_row)

        deq_init = run_pnp_variant(
            args,
            problem,
            denoiser,
            n_iter=args.deqInitIter,
            alpha=0.0,
            init_name="DEQ-MPI",
            x0=deq_mpi,
        )
        refine_metrics, refine_sample_psnr = method_metrics(
            "DEQ-init ZeroShot-PnP",
            "DEQ-MPI",
            args.deqInitIter,
            0.0,
            deq_init["x_rec"],
            deq_init["elapsed_seconds"],
            ref_vals,
            problem,
        )
        refine_row = add_common_fields(
            refine_metrics,
            args,
            problem,
            checkpoint=checkpoint,
            epoch_label=label,
            training_regime="supervised checkpoint plus fixed zero-shot refinement",
        )
        rows.append(refine_row)

        payload = {
            "seed": args.seed,
            "pSNRval": args.pSNRval,
            "num_test_images": problem["num_test_images"],
            "epoch_label": label,
            "checkpoint": str(checkpoint),
            "denoiser": args.denoiser,
            "deq_mpi": deq_row,
            "deq_init_zeroshot_pnp": refine_row,
        }
        json_path = out_dir / f"checkpoint_epoch{label}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        deq_row["json"] = str(json_path)
        refine_row["json"] = str(json_path)

        for method, values in [("DEQ-MPI", deq_sample_psnr), ("DEQ-init ZeroShot-PnP", refine_sample_psnr)]:
            for idx, value in enumerate(values):
                sample_rows.append(
                    {
                        "epoch_label": label,
                        "epoch": epoch_label_to_x(label),
                        "method": method,
                        "sample_index": idx,
                        "psnr_db": float(value),
                    }
                )

    rows = sorted(rows, key=lambda row: (int(row["epoch"]), row["method"]))
    write_summary_csv(out_csv, rows)
    with (out_dir / "few_training_all_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch_label", "epoch", "method", "sample_index", "psnr_db"])
        writer.writeheader()
        writer.writerows(sample_rows)
    save_curve(rows, figure_path)

    best_deq = max([row for row in rows if row["method"] == "DEQ-MPI"], key=lambda row: float(row["mean_psnr_db"]))
    best_refine = max(
        [row for row in rows if row["method"] == "DEQ-init ZeroShot-PnP"],
        key=lambda row: float(row["mean_psnr_db"]),
    )
    manifest = {
        "seed": args.seed,
        "pSNRval": args.pSNRval,
        "num_test_images": problem["num_test_images"],
        "denoiser": args.denoiser,
        "checkpointDir": args.checkpointDir,
        "epochs": parse_epoch_labels(args.epochs),
        "elapsed_seconds": time.time() - start_time,
        "zero_training": zero_row,
        "best_deq_mpi": best_deq,
        "best_deq_initialized_refinement": best_refine,
        "outputs": {
            "summary_csv": str(out_csv),
            "all_sample_metrics_csv": str(out_dir / "few_training_all_sample_metrics.csv"),
            "curve_png": str(figure_path),
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
