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
import torch.nn.functional as F
from scipy.io import loadmat

from data import MRAdatasetH5NoScale, loadMtxExp
from modelClasses import DEQFixedPoint, fixedPointTekrarlar, rdnLDFixedPt
from reconAlgos import ADMMfncs, psnr
from trainerClasses import admmInputGenerator, getNoisyData, transformDataset
from zeroshot_l1_pnp import ZeroShotL1PnP, build_denoiser


DEFAULT_CHECKPOINT = (
    "training/reproduction/deqmpi/"
    "DeqMPI_1D_ds_lr_0.001_wd_0_bs_64_pSNR_10.0_fixNs_1_Nit_5_"
    "nF12_nB4_lieb4_gr12_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_"
    "LnF_8_LnB_1_nN_1/epoch200END.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create 2D paper figures comparing DEQ-MPI and ZeroShot-l1-PnP "
            "on the same simulated MPI test samples."
        )
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

    parser.add_argument("--nIterPnp", type=int, default=7)
    parser.add_argument("--mu0", type=float, default=5e5)
    parser.add_argument("--pnpAlpha", type=float, default=0.0)
    parser.add_argument("--l1Alpha", type=float, default=None)
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
        help="Automatic sample selection rule when sampleIndices is not set.",
    )
    parser.add_argument(
        "--minGtEnergyQuantile",
        type=float,
        default=0.5,
        help="Ground-truth energy quantile used by gt_energy_filtered_deq_quantiles.",
    )
    parser.add_argument(
        "--sampleIndices",
        default="",
        help="Optional comma-separated absolute test-set indices. Overrides quantile selection.",
    )
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--outDir", default="training/reproduction/paper_figures")
    parser.add_argument("--prefix", default="deqmpi_vs_zeroshot_2d")
    return parser.parse_args()


def setup_device(gpu: int):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This reproduction path requires CUDA because the original data loader "
            "and system-matrix utilities move tensors to cuda()."
        )
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def build_problem(args, device):
    n1, n2 = 26, 13
    img_size = [n1, n2]
    n_img = n1 * n2

    sys_mtx_ref = loadMtxExp("inhouseData/expMatinHouse").reshape(-1, n_img)
    interpolater = loadmat("interpExp2.mat")["interpolater"]
    sys_mtx_hr_int2 = sys_mtx_ref @ torch.from_numpy(interpolater).float().to(device)
    sys_mtx = F.avg_pool2d(
        sys_mtx_hr_int2.reshape(sys_mtx_ref.shape[0], 2 * n1, 2 * n2), 2
    ).reshape(sys_mtx_ref.shape[0], -1)

    u, s, vh = torch.linalg.svd(sys_mtx, full_matrices=False)
    u = u[:, : args.nbOfSingulars]
    s = s[: args.nbOfSingulars]
    v = vh[: args.nbOfSingulars, :].T
    a_reduced = torch.diag(s) @ v.T

    test_data_hr = MRAdatasetH5NoScale("datasets/testPatches.h5", prefetch=True).data
    test_data = transformDataset(test_data_hr, img_size, [0.5, 1], [0, 0])
    if args.testLimit and args.testLimit > 0:
        test_data = test_data[: args.testLimit]

    ref_vals = test_data.squeeze().reshape(test_data.shape[0], -1)

    my_data_nsless = getNoisyData(test_data, 0, sys_mtx_ref)
    my_data_lr_inv_cr = getNoisyData(test_data, 0, sys_mtx)
    std_val = 10 ** (-args.pSNRval / 20) * 0.41
    my_data_gen = getNoisyData(test_data, std_val, sys_mtx_ref)

    measured_snr_no_inverse_crime = float(
        20 * torch.log10(torch.norm(my_data_nsless) / torch.norm(my_data_gen - my_data_nsless))
    )
    measured_snr_inverse_crime = float(
        20 * torch.log10(torch.norm(my_data_lr_inv_cr) / torch.norm(my_data_gen - my_data_lr_inv_cr))
    )

    datat_c_full = my_data_gen.reshape(test_data.shape[0], -1)
    _comp_data, lsqr_inp = admmInputGenerator(my_data_gen, u, s, v, (-1, 1, n1, n2))
    y_reduced = datat_c_full @ u

    at_c = sys_mtx.reshape(-1, n_img)
    admm = ADMMfncs(at_c, args.maxIter, 100, img_size, diagnose=False)

    comp_for_ls = F.linear(datat_c_full.reshape(test_data.shape[0], -1), u.T)
    val_inp = F.linear(comp_for_ls / (s + 1e-4), v).reshape(-1, 1, n1, n2)

    return {
        "n1": n1,
        "n2": n2,
        "n_img": n_img,
        "img_size": img_size,
        "sys_mtx": sys_mtx,
        "u": u,
        "s": s,
        "v": v,
        "A_reduced": a_reduced,
        "ref_vals": ref_vals,
        "datat_c_full": datat_c_full,
        "y_reduced": y_reduced,
        "lsqr": lsqr_inp.reshape(test_data.shape[0], -1),
        "at_c": at_c,
        "madmm": admm.MtC,
        "epsilon": float(std_val * (my_data_nsless.shape[2]) ** 0.5),
        "measured_snr_no_inverse_crime_db": measured_snr_no_inverse_crime,
        "measured_snr_inverse_crime_db": measured_snr_inverse_crime,
        "num_test_images": int(test_data.shape[0]),
    }


def run_deq(args, problem, device):
    n1, n2, n_img = problem["n1"], problem["n2"], problem["n_img"]
    datat_c_full = problem["datat_c_full"]
    ref_vals = problem["ref_vals"]
    val_inp = F.linear(
        F.linear(datat_c_full.reshape(datat_c_full.shape[0], -1), problem["u"].T)
        / (problem["s"] + 1e-4),
        problem["v"],
    ).reshape(-1, 1, n1, n2)

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
    model = DEQFixedPoint(model_core, fixedPointTekrarlar, max_iter=args.maxIter).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    n_batch = datat_c_full.shape[0]
    d0 = torch.zeros_like(val_inp).reshape(n_batch, -1)
    d2 = torch.zeros_like(datat_c_full)
    model_in = torch.cat((val_inp.reshape(n_batch, -1), d0, d2), dim=1)
    x_rec = torch.zeros_like(ref_vals).reshape(-1, 1, n1, n2)

    start_time = time.time()
    with torch.no_grad():
        for start in range(0, n_batch, args.batchSizeDeq):
            end = min(start + args.batchSizeDeq, n_batch)
            batch_shape = (end - start, 1, n1, n2)
            fixed_pts = (
                datat_c_full[start:end],
                problem["at_c"],
                problem["madmm"],
                problem["epsilon"],
                batch_shape,
            )
            x_rec[start:end] = model(model_in[start:end], fixed_pts)[:, :n_img].reshape(batch_shape)
    return x_rec.reshape(n_batch, -1), time.time() - start_time


def run_pnp(args, problem, device, denoiser, alpha):
    reconstructor = ZeroShotL1PnP(
        problem["A_reduced"].to(device),
        problem["img_size"],
        denoiser,
        n_iter=args.nIterPnp,
        mu0=args.mu0,
        alpha=alpha,
        direct_solve=True,
        non_negative=True,
    )

    y_reduced = problem["y_reduced"]
    x_rec = torch.zeros(y_reduced.shape[0], problem["n_img"], device=device, dtype=y_reduced.dtype)
    histories = []
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, y_reduced.shape[0], args.batchSizePnp):
            end = min(start + args.batchSizePnp, y_reduced.shape[0])
            out, hist = reconstructor(y_reduced[start:end], return_history=True)
            x_rec[start:end] = out
            if not histories:
                histories = hist
    elapsed = time.time() - start_time
    return x_rec, elapsed, histories, reconstructor.alpha


def summarize(name, values, elapsed):
    return {
        "method": name,
        "mean_psnr_db": float(np.mean(values)),
        "std_psnr_db": float(np.std(values)),
        "min_psnr_db": float(np.min(values)),
        "max_psnr_db": float(np.max(values)),
        "elapsed_seconds": float(elapsed),
    }


def select_examples(args, deq_psnr, ref_vals):
    num_images = ref_vals.shape[0]
    if args.sampleIndices.strip():
        indices = [int(v.strip()) for v in args.sampleIndices.split(",") if v.strip()]
        bad = [idx for idx in indices if idx < 0 or idx >= num_images]
        if bad:
            raise ValueError(f"sample indices out of range for {num_images} images: {bad}")
        return indices

    n = min(args.numExamples, num_images)
    candidates = np.arange(num_images)
    if args.selection == "gt_energy_filtered_deq_quantiles":
        energy = ref_vals.detach().cpu().numpy().sum(axis=1)
        threshold = np.quantile(energy, args.minGtEnergyQuantile)
        candidates = np.flatnonzero(energy >= threshold)
        if len(candidates) < n:
            candidates = np.arange(num_images)

    quantiles = np.linspace(0.1, 0.9, n)
    selected = []
    for q in quantiles:
        candidate_psnr = deq_psnr[candidates]
        target = np.quantile(candidate_psnr, q)
        order = candidates[np.argsort(np.abs(candidate_psnr - target))]
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
                break
    return selected


def tensor_to_images(x_flat, n1, n2):
    return x_flat.detach().cpu().numpy().reshape(-1, n1, n2)


def save_panel(args, problem, outputs, psnr_by_method, selected_indices, out_path):
    n1, n2 = problem["n1"], problem["n2"]
    method_order = [
        ("Ground truth", "ground_truth"),
        ("LS/SVD", "ls_svd"),
        ("DEQ-MPI", "deq_mpi"),
        ("ZeroShot-PnP", "zeroshot_pnp"),
        ("ZeroShot-l1-PnP", "zeroshot_l1_pnp"),
    ]
    images = {key: tensor_to_images(value, n1, n2) for key, value in outputs.items()}

    n_rows = len(selected_indices)
    n_cols = len(method_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols, 1.55 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, sample_idx in enumerate(selected_indices):
        row_imgs = [np.clip(images[key][sample_idx], 0, None) for _title, key in method_order]
        vmax = max(float(np.max(img)) for img in row_imgs)
        vmax = max(vmax, 1e-6)

        for col, (title, key) in enumerate(method_order):
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
                ax.set_title(title, fontsize=10, pad=8)
            if col == 0:
                ax.text(
                    -0.16,
                    0.5,
                    f"idx {sample_idx}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            if key == "ground_truth":
                label = "reference"
            else:
                label = f"{psnr_by_method[key][sample_idx]:.1f} dB"
            ax.text(
                0.5,
                -0.12,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )

    fig.suptitle(
        "2D simulated MPI examples: DEQ-MPI baseline and zero-shot prior variants",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metric_rows(out_csv, psnr_by_method, selected_indices):
    rows = []
    for method_key, values in psnr_by_method.items():
        if method_key == "ground_truth":
            continue
        for idx, value in enumerate(values):
            rows.append(
                {
                    "sample_index": idx,
                    "method": method_key,
                    "psnr_db": float(value),
                    "selected_for_figure": int(idx in selected_indices),
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_index", "method", "psnr_db", "selected_for_figure"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_selected_rows(out_csv, psnr_by_method, selected_indices):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "ls_svd_psnr_db",
                "deq_mpi_psnr_db",
                "zeroshot_pnp_psnr_db",
                "zeroshot_l1_pnp_psnr_db",
            ],
        )
        writer.writeheader()
        for idx in selected_indices:
            writer.writerow(
                {
                    "sample_index": idx,
                    "ls_svd_psnr_db": float(psnr_by_method["ls_svd"][idx]),
                    "deq_mpi_psnr_db": float(psnr_by_method["deq_mpi"][idx]),
                    "zeroshot_pnp_psnr_db": float(psnr_by_method["zeroshot_pnp"][idx]),
                    "zeroshot_l1_pnp_psnr_db": float(psnr_by_method["zeroshot_l1_pnp"][idx]),
                }
            )


def main():
    args = parse_args()
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    zeroshot_pnp, pnp_elapsed, pnp_history, pnp_alpha = run_pnp(
        args, problem, device, denoiser, args.pnpAlpha
    )
    zeroshot_l1_pnp, l1_elapsed, l1_history, l1_alpha = run_pnp(
        args, problem, device, denoiser, args.l1Alpha
    )

    psnr_by_method = {
        "ls_svd": psnr(ref_vals, ls_svd),
        "deq_mpi": psnr(ref_vals, deq_mpi),
        "zeroshot_pnp": psnr(ref_vals, zeroshot_pnp),
        "zeroshot_l1_pnp": psnr(ref_vals, zeroshot_l1_pnp),
    }

    selected_indices = select_examples(args, psnr_by_method["deq_mpi"], ref_vals)

    outputs = {
        "ground_truth": ref_vals,
        "ls_svd": ls_svd,
        "deq_mpi": deq_mpi,
        "zeroshot_pnp": zeroshot_pnp,
        "zeroshot_l1_pnp": zeroshot_l1_pnp,
    }

    panel_path = out_dir / f"{args.prefix}_examples.png"
    save_panel(args, problem, outputs, psnr_by_method, selected_indices, panel_path)

    metrics_csv = out_dir / f"{args.prefix}_all_sample_metrics.csv"
    selected_csv = out_dir / f"{args.prefix}_selected_examples.csv"
    summary_json = out_dir / f"{args.prefix}_summary.json"
    write_metric_rows(metrics_csv, psnr_by_method, selected_indices)
    write_selected_rows(selected_csv, psnr_by_method, selected_indices)

    summary = {
        "seed": args.seed,
        "num_test_images": problem["num_test_images"],
        "pSNRval": args.pSNRval,
        "measured_snr_no_inverse_crime_db": problem["measured_snr_no_inverse_crime_db"],
        "measured_snr_inverse_crime_db": problem["measured_snr_inverse_crime_db"],
        "nbOfSingulars": args.nbOfSingulars,
        "maxIter_deq": args.maxIter,
        "nIter_pnp": args.nIterPnp,
        "mu0": args.mu0,
        "zeroshot_pnp_alpha": pnp_alpha,
        "zeroshot_l1_pnp_alpha": l1_alpha,
        "denoiser": args.denoiser,
        "checkpoint": args.checkpoint,
        "selected_indices": selected_indices,
        "selection_rule": (
            "explicit sampleIndices" if args.sampleIndices.strip() else args.selection
        ),
        "minGtEnergyQuantile": args.minGtEnergyQuantile,
        "outputs": {
            "panel_png": str(panel_path),
            "all_sample_metrics_csv": str(metrics_csv),
            "selected_examples_csv": str(selected_csv),
        },
        "first_batch_histories": {
            "zeroshot_pnp": pnp_history,
            "zeroshot_l1_pnp": l1_history,
        },
        "summary": [
            summarize("LS/SVD", psnr_by_method["ls_svd"], 0.0),
            summarize("DEQ-MPI", psnr_by_method["deq_mpi"], deq_elapsed),
            summarize("ZeroShot-PnP", psnr_by_method["zeroshot_pnp"], pnp_elapsed),
            summarize("ZeroShot-l1-PnP", psnr_by_method["zeroshot_l1_pnp"], l1_elapsed),
        ],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
