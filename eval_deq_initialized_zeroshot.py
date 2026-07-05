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

from make_2d_paper_comparison import (
    DEFAULT_CHECKPOINT,
    build_problem,
    run_deq,
    select_examples,
    setup_device,
    tensor_to_images,
)
from reconAlgos import psnr
from zeroshot_l1_pnp import ZeroShotL1PnP, build_denoiser


def parse_csv_ints(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_floats(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DEQ-initialized ZeroShot-PnP refinement on the 2D DEQ-MPI benchmark."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--maxIter", type=int, default=25)
    parser.add_argument("--batchSizeDeq", type=int, default=256)
    parser.add_argument("--batchSizePnp", type=int, default=64)
    parser.add_argument("--testLimit", type=int, default=0)

    parser.add_argument("--zeroInitIters", default="7")
    parser.add_argument("--deqInitPnpIters", default="1,3,5")
    parser.add_argument("--deqInitL1Iters", default="1,3")
    parser.add_argument("--l1Alphas", default="500,1000,2500")
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

    parser.add_argument("--outDir", default="training/reproduction/deq_init_zeroshot")
    parser.add_argument("--outCsv", default="training/reproduction/deq_init_zeroshot_summary.csv")
    parser.add_argument(
        "--figurePath",
        default="training/reproduction/paper_figures/deq_initialized_zeroshot_examples.png",
    )
    return parser.parse_args()


def run_pnp_variant(args, problem, denoiser, *, n_iter, alpha, init_name, x0):
    reconstructor = ZeroShotL1PnP(
        problem["A_reduced"],
        problem["img_size"],
        denoiser,
        n_iter=n_iter,
        mu0=args.mu0,
        alpha=alpha,
        direct_solve=args.directSolve,
        non_negative=True,
    )

    y_reduced = problem["y_reduced"]
    x_rec = torch.zeros(y_reduced.shape[0], problem["n_img"], device=y_reduced.device, dtype=y_reduced.dtype)
    histories = []
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, y_reduced.shape[0], args.batchSizePnp):
            end = min(start + args.batchSizePnp, y_reduced.shape[0])
            x0_batch = None if x0 is None else x0[start:end]
            out, hist = reconstructor(y_reduced[start:end], return_history=True, x0=x0_batch)
            x_rec[start:end] = out
            if not histories:
                histories = hist

    return {
        "key": make_variant_key(init_name, n_iter, reconstructor.alpha),
        "init": init_name,
        "n_iter": int(n_iter),
        "alpha": float(reconstructor.alpha),
        "x_rec": x_rec,
        "elapsed_seconds": time.time() - start_time,
        "first_batch_history": histories,
    }


def make_variant_key(init_name, n_iter, alpha):
    alpha_tag = f"{float(alpha):g}".replace(".", "p")
    return f"{init_name.lower()}_init_iter{int(n_iter)}_alpha{alpha_tag}"


def summarize_variant(variant, ref_vals, problem):
    values = psnr(ref_vals, variant["x_rec"])
    residual = torch.linalg.norm(problem["y_reduced"] - variant["x_rec"] @ problem["A_reduced"].T, dim=1)
    nrmse = torch.linalg.norm(ref_vals - variant["x_rec"], dim=1) / torch.linalg.norm(ref_vals, dim=1).clamp_min(1e-12)
    nonzero_ratio = (variant["x_rec"].abs() > 1e-8).float().mean(dim=1)
    return {
        "key": variant["key"],
        "method": "ZeroShot-PnP" if variant["alpha"] == 0 else "ZeroShot-l1-PnP",
        "init": variant["init"],
        "n_iter": variant["n_iter"],
        "alpha": variant["alpha"],
        "mean_psnr_db": float(np.mean(values)),
        "std_psnr_db": float(np.std(values)),
        "min_psnr_db": float(np.min(values)),
        "max_psnr_db": float(np.max(values)),
        "mean_nrmse": float(nrmse.mean().detach().cpu()),
        "mean_residual_norm": float(residual.mean().detach().cpu()),
        "mean_nonzero_ratio": float(nonzero_ratio.mean().detach().cpu()),
        "elapsed_seconds": float(variant["elapsed_seconds"]),
    }, values


def write_variant_json(path, args, problem, variant, summary, sample_psnr):
    payload = {
        "method": summary["method"],
        "init": summary["init"],
        "seed": args.seed,
        "num_test_images": problem["num_test_images"],
        "pSNRval": args.pSNRval,
        "measured_snr_no_inverse_crime_db": problem["measured_snr_no_inverse_crime_db"],
        "measured_snr_inverse_crime_db": problem["measured_snr_inverse_crime_db"],
        "nbOfSingulars": args.nbOfSingulars,
        "nIter": summary["n_iter"],
        "mu0": args.mu0,
        "alpha": summary["alpha"],
        "denoiser": args.denoiser,
        "checkpoint": args.checkpoint,
        "mean_psnr_db": summary["mean_psnr_db"],
        "std_psnr_db": summary["std_psnr_db"],
        "min_psnr_db": summary["min_psnr_db"],
        "max_psnr_db": summary["max_psnr_db"],
        "mean_nrmse": summary["mean_nrmse"],
        "mean_residual_norm": summary["mean_residual_norm"],
        "mean_nonzero_ratio": summary["mean_nonzero_ratio"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "first_batch_history": variant["first_batch_history"],
        "sample_psnr_db": [float(v) for v in sample_psnr],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary_csv(path, rows):
    fieldnames = [
        "rank",
        "key",
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
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)


def write_sample_metrics_csv(path, sample_rows, psnr_by_key, selected_indices):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_index", "method", "init", "n_iter", "alpha", "psnr_db", "selected_for_figure"],
        )
        writer.writeheader()
        for row in sample_rows:
            for idx, value in enumerate(psnr_by_key[row["key"]]):
                writer.writerow(
                    {
                        "sample_index": idx,
                        "method": row["method"],
                        "init": row["init"],
                        "n_iter": row["n_iter"],
                        "alpha": row["alpha"],
                        "psnr_db": float(value),
                        "selected_for_figure": int(idx in selected_indices),
                    }
                )


def save_refinement_panel(
    args,
    problem,
    outputs,
    psnr_by_key,
    selected_indices,
    out_path,
    title="DEQ-initialized zero-shot refinement on 2D simulated MPI",
):
    n1, n2 = problem["n1"], problem["n2"]
    method_order = [(title, key) for title, key in outputs["order"]]
    images = {key: tensor_to_images(value, n1, n2) for key, value in outputs["images"].items()}

    n_rows = len(selected_indices)
    n_cols = len(method_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.25 * n_cols, 1.55 * n_rows),
        squeeze=False,
    )

    for row, sample_idx in enumerate(selected_indices):
        row_imgs = [np.clip(images[key][sample_idx], 0, None) for _title, key in method_order]
        vmax = max(max(float(np.max(img)) for img in row_imgs), 1e-6)

        for col, (method_title, key) in enumerate(method_order):
            ax = axes[row][col]
            ax.imshow(
                np.clip(images[key][sample_idx], 0, None),
                cmap=args.cmap,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method_title, fontsize=9, pad=8)
            if col == 0:
                ax.text(-0.16, 0.5, f"idx {sample_idx}", transform=ax.transAxes, ha="right", va="center", fontsize=8)
            label = "reference" if key == "ground_truth" else f"{psnr_by_key[key][sample_idx]:.1f} dB"
            ax.text(0.5, -0.12, label, transform=ax.transAxes, ha="center", va="top", fontsize=8)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    ls_svd = problem["lsqr"]
    deq_mpi, deq_elapsed = run_deq(args, problem, device)

    denoiser_args = SimpleNamespace(
        denoiser=args.denoiser,
        gaussianKernel=args.gaussianKernel,
        gaussianSigma=args.gaussianSigma,
        dpirRoot=args.dpirRoot,
        drunetWeights=args.drunetWeights,
    )
    denoiser = build_denoiser(denoiser_args, device)

    variants = []
    for n_iter in parse_csv_ints(args.zeroInitIters):
        variants.append(run_pnp_variant(args, problem, denoiser, n_iter=n_iter, alpha=0.0, init_name="zero", x0=None))
    for n_iter in parse_csv_ints(args.deqInitPnpIters):
        variants.append(run_pnp_variant(args, problem, denoiser, n_iter=n_iter, alpha=0.0, init_name="DEQ-MPI", x0=deq_mpi))
    for n_iter in parse_csv_ints(args.deqInitL1Iters):
        for alpha in parse_csv_floats(args.l1Alphas):
            variants.append(run_pnp_variant(args, problem, denoiser, n_iter=n_iter, alpha=alpha, init_name="DEQ-MPI", x0=deq_mpi))

    psnr_by_key = {
        "ls_svd": psnr(ref_vals, ls_svd),
        "deq_mpi": psnr(ref_vals, deq_mpi),
    }
    rows = [
        {
            "key": "ls_svd",
            "method": "LS/SVD",
            "init": "none",
            "n_iter": 0,
            "alpha": 0.0,
            "mean_psnr_db": float(np.mean(psnr_by_key["ls_svd"])),
            "std_psnr_db": float(np.std(psnr_by_key["ls_svd"])),
            "min_psnr_db": float(np.min(psnr_by_key["ls_svd"])),
            "max_psnr_db": float(np.max(psnr_by_key["ls_svd"])),
            "mean_nrmse": "",
            "mean_residual_norm": "",
            "mean_nonzero_ratio": "",
            "elapsed_seconds": 0.0,
            "json": "",
        },
        {
            "key": "deq_mpi",
            "method": "DEQ-MPI",
            "init": "learned",
            "n_iter": args.maxIter,
            "alpha": 0.0,
            "mean_psnr_db": float(np.mean(psnr_by_key["deq_mpi"])),
            "std_psnr_db": float(np.std(psnr_by_key["deq_mpi"])),
            "min_psnr_db": float(np.min(psnr_by_key["deq_mpi"])),
            "max_psnr_db": float(np.max(psnr_by_key["deq_mpi"])),
            "mean_nrmse": "",
            "mean_residual_norm": "",
            "mean_nonzero_ratio": "",
            "elapsed_seconds": float(deq_elapsed),
            "json": "",
        },
    ]

    variant_by_key = {}
    for variant in variants:
        summary, sample_psnr = summarize_variant(variant, ref_vals, problem)
        json_path = out_dir / f"{summary['key']}.json"
        write_variant_json(json_path, args, problem, variant, summary, sample_psnr)
        summary["json"] = str(json_path)
        rows.append(summary)
        psnr_by_key[summary["key"]] = sample_psnr
        variant_by_key[summary["key"]] = variant

    rows_sorted = sorted(rows, key=lambda row: row["mean_psnr_db"], reverse=True)
    write_summary_csv(out_csv, rows_sorted)

    selected_indices = select_examples(args, psnr_by_key["deq_mpi"], ref_vals)
    write_sample_metrics_csv(out_dir / "all_sample_metrics.csv", rows, psnr_by_key, selected_indices)

    deq_init_pnp = [row for row in rows if row["method"] == "ZeroShot-PnP" and row["init"] == "DEQ-MPI"]
    deq_init_l1 = [row for row in rows if row["method"] == "ZeroShot-l1-PnP" and row["init"] == "DEQ-MPI"]
    best_pnp = max(deq_init_pnp, key=lambda row: row["mean_psnr_db"]) if deq_init_pnp else None
    best_l1 = max(deq_init_l1, key=lambda row: row["mean_psnr_db"]) if deq_init_l1 else None
    zero_row = next(row for row in rows if row["method"] == "ZeroShot-PnP" and row["init"] == "zero")

    output_images = {
        "ground_truth": ref_vals,
        "deq_mpi": deq_mpi,
        zero_row["key"]: variant_by_key[zero_row["key"]]["x_rec"],
    }
    output_order = [
        ("Ground truth", "ground_truth"),
        ("DEQ-MPI", "deq_mpi"),
        (f"ZeroShot {zero_row['n_iter']} it", zero_row["key"]),
    ]
    if best_pnp is not None:
        output_images[best_pnp["key"]] = variant_by_key[best_pnp["key"]]["x_rec"]
        output_order.append((f"DEQ-init PnP {best_pnp['n_iter']} it", best_pnp["key"]))
    if best_l1 is not None:
        output_images[best_l1["key"]] = variant_by_key[best_l1["key"]]["x_rec"]
        output_order.append((f"DEQ-init l1 {best_l1['n_iter']} it", best_l1["key"]))

    save_refinement_panel(
        args,
        problem,
        {"images": output_images, "order": output_order},
        psnr_by_key,
        selected_indices,
        figure_path,
    )

    manifest = {
        "seed": args.seed,
        "num_test_images": problem["num_test_images"],
        "pSNRval": args.pSNRval,
        "denoiser": args.denoiser,
        "selected_indices": selected_indices,
        "outputs": {
            "summary_csv": str(out_csv),
            "all_sample_metrics_csv": str(out_dir / "all_sample_metrics.csv"),
            "figure_png": str(figure_path),
        },
        "best_by_mean_psnr": rows_sorted[0],
        "deq_mpi": next(row for row in rows if row["key"] == "deq_mpi"),
        "best_deq_initialized_pnp": best_pnp,
        "best_deq_initialized_l1_pnp": best_l1,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
