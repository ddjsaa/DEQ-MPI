"""Train or smoke-test Sparse-PnP-ADMM-DEQ on the 2D MPI benchmark.

Examples:
  python train_sparse_pnp_admm_deq.py --trainLimit 64 --valLimit 64 --epochs 1 --solver finite
  python train_sparse_pnp_admm_deq.py --trainLimit 8 --valLimit 8 --epochs 1 --solver deq --maxIter 3

Outputs are written under training/reproduction/sparse_pnp_admm_deq/<prefix>/:
  - args.json
  - training_log.csv
  - summary.json
  - epoch* checkpoint files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import MRAdatasetH5NoScale, loadMtxExp
from modelClasses import rdnDenoiserResRelu
from sparse_pnp_admm_deq import SparsePnPADMMDEQ, SparsePnPADMMFixedPoint
from trainerClasses import getNoisyData, transformDataset
from zeroshot_l1_pnp import GaussianDenoiser


DEFAULT_DENOISER = (
    "training/reproduction/denoiser/"
    "ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_"
    "nF12_nB4_lieb4_gr12_rMn0.5_1.0/epoch200END.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse-PnP-ADMM-DEQ training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trainPath", default="datasets/trainPatches.h5")
    parser.add_argument("--valPath", default="datasets/valPatches.h5")
    parser.add_argument("--trainLimit", type=int, default=0)
    parser.add_argument("--valLimit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--valBatchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--rescaleMin", type=float, default=0.5)
    parser.add_argument("--rescaleMax", type=float, default=1.0)
    parser.add_argument("--solver", choices=["finite", "deq"], default="finite")
    parser.add_argument("--deqSolver", choices=["fixed", "regular", "anderson"], default="fixed")
    parser.add_argument("--maxIter", type=int, default=5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--l1Lambda", type=float, default=0.0005)
    parser.add_argument("--allowNegative", action="store_true")
    parser.add_argument("--denoiser", choices=["identity", "gaussian", "rdn"], default="rdn")
    parser.add_argument("--gaussianKernel", type=int, default=3)
    parser.add_argument("--gaussianSigma", type=float, default=0.65)
    parser.add_argument("--preloadDenoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--freezeDenoiser", action="store_true")
    parser.add_argument("--nb_of_features", type=int, default=12)
    parser.add_argument("--nb_of_blocks", type=int, default=4)
    parser.add_argument("--layer_in_each_block", type=int, default=4)
    parser.add_argument("--growth_rate", type=int, default=12)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--saveEvery", type=int, default=0)
    parser.add_argument("--outDir", default="training/reproduction/sparse_pnp_admm_deq")
    parser.add_argument("--prefix", default="train_smoke")
    parser.add_argument("--detectAnomaly", action="store_true")
    parser.add_argument("--strictSmoke", action="store_true")
    return parser.parse_args()


def setup_device(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because repository matrix loaders call cuda().")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def load_tensor_dataset(path: str, limit: int) -> TensorDataset:
    data = MRAdatasetH5NoScale(path, prefetch=True, dim=2, device=torch.device("cpu")).data
    if limit and limit > 0:
        data = data[:limit].clone()
    return TensorDataset(data)


def build_system(device: torch.device, nb_singulars: int):
    n1, n2 = 26, 13
    n_img = n1 * n2
    sys_mtx = loadMtxExp("inhouseData/expMatinHouse").reshape(-1, n_img).to(device)
    u, s, vh = torch.linalg.svd(sys_mtx, full_matrices=False)
    u = u[:, :nb_singulars]
    s = s[:nb_singulars]
    v = vh[:nb_singulars, :].T
    a_reduced = torch.diag(s) @ v.T
    return {
        "img_size": [n1, n2],
        "n_img": n_img,
        "sys_mtx": sys_mtx,
        "u": u,
        "s": s,
        "v": v,
        "a_reduced": a_reduced,
    }


def make_denoiser(args, device: torch.device) -> nn.Module:
    if args.denoiser == "identity":
        denoiser = nn.Identity().to(device)
    elif args.denoiser == "gaussian":
        denoiser = GaussianDenoiser(args.gaussianKernel, args.gaussianSigma).to(device)
    else:
        denoiser = rdnDenoiserResRelu(
            input_channels=1,
            nb_of_features=args.nb_of_features,
            nb_of_blocks=args.nb_of_blocks,
            layer_in_each_block=args.layer_in_each_block,
            growth_rate=args.growth_rate,
            out_channel=1,
            bias=True,
        ).to(device)
        if args.preloadDenoiser:
            checkpoint = Path(args.preloadDenoiser)
            if checkpoint.exists():
                denoiser.load_state_dict(torch.load(checkpoint, map_location=device))
            else:
                raise FileNotFoundError(f"pretrained denoiser not found: {checkpoint}")

    if args.freezeDenoiser:
        for param in denoiser.parameters():
            param.requires_grad = False
        denoiser.eval()
    return denoiser


def prepare_batch(batch, system, args, device: torch.device, train: bool):
    data_hr = batch[0].to(device).float()
    rand_vals = None if train else [0, 0]
    data = transformDataset(
        data_hr.clone(),
        system["img_size"],
        [args.rescaleMin, args.rescaleMax],
        rand_vals,
    )
    std_val = 0.41 * 10 ** (-args.pSNRval / 20)
    noisy_full = getNoisyData(data, std_val, system["sys_mtx"])
    y_reduced = F.linear(noisy_full.reshape(data.shape[0], -1), system["u"].T)
    x0 = F.linear(y_reduced / (system["s"] + 1e-4), system["v"]).reshape(data.shape[0], -1)
    target = data.reshape(data.shape[0], -1)
    return y_reduced, x0, target


def psnr_mean(target: torch.Tensor, pred: torch.Tensor) -> float:
    n_pix = target.shape[1]
    err = torch.linalg.norm(target - pred, dim=1).clamp_min(1e-12)
    peak = target.max().clamp_min(1e-12)
    vals = 10 * torch.log10(n_pix * peak.square() / err.square())
    return float(vals.mean().detach().cpu())


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().norm().cpu()) ** 2
    return math.sqrt(total)


def run_one_epoch(model, loader, system, args, device, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    rows = []
    total_loss = 0.0
    total_count = 0
    total_nrmse_num = 0.0
    total_nrmse_den = 0.0
    total_psnr_weighted = 0.0
    total_grad = 0.0
    grad_steps = 0
    start_time = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            y_reduced, x0, target = prepare_batch(batch, system, args, device, train=train)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(y_reduced, x0=x0)
            loss = loss_fn(pred, target)
            if train:
                loss.backward()
                total_grad += grad_norm(model.parameters())
                grad_steps += 1
                optimizer.step()

            count = target.shape[0]
            nrmse_num = float(torch.linalg.norm(target - pred).detach().cpu()) ** 2
            nrmse_den = float(torch.linalg.norm(target).detach().cpu()) ** 2
            batch_psnr = psnr_mean(target, pred)
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            total_nrmse_num += nrmse_num
            total_nrmse_den += nrmse_den
            total_psnr_weighted += batch_psnr * count
            rows.append(
                {
                    "batch": batch_idx,
                    "loss": float(loss.detach().cpu()),
                    "psnr_db": batch_psnr,
                    "nrmse": math.sqrt(nrmse_num / max(nrmse_den, 1e-24)),
                }
            )

    return {
        "loss": total_loss / max(total_count, 1),
        "nrmse": math.sqrt(total_nrmse_num / max(total_nrmse_den, 1e-24)),
        "psnr_db": total_psnr_weighted / max(total_count, 1),
        "elapsed_seconds": time.time() - start_time,
        "grad_norm": total_grad / max(grad_steps, 1) if train else None,
        "batches": rows,
    }


def write_log(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "split",
                "loss",
                "nrmse",
                "psnr_db",
                "elapsed_seconds",
                "grad_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.autograd.set_detect_anomaly(args.detectAnomaly)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = setup_device(args.gpu)

    save_dir = Path(args.outDir) / args.prefix
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    system = build_system(device, args.nbOfSingulars)
    train_dataset = load_tensor_dataset(args.trainPath, args.trainLimit)
    val_dataset = load_tensor_dataset(args.valPath, args.valLimit)
    train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.valBatchSize, shuffle=False)

    denoiser = make_denoiser(args, device)
    fixed_point = SparsePnPADMMFixedPoint(
        A=system["a_reduced"],
        image_shape=system["img_size"],
        denoiser=denoiser,
        rho=args.rho,
        eta=args.eta,
        l1_lambda=args.l1Lambda,
        non_negative=not args.allowNegative,
    ).to(device)
    model = SparsePnPADMMDEQ(
        fixed_point,
        max_iter=args.maxIter,
        solver=args.solver,
        deq_solver=args.deqSolver,
        solver_kwargs={"tol": 1e-4, "beta": 1.0} if args.deqSolver in ("regular", "anderson") else None,
    ).to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Remove --freezeDenoiser or use a trainable denoiser.")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()

    log_rows = []
    best_val = -float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, system, args, device, loss_fn, optimizer)
        val_metrics = run_one_epoch(model, val_loader, system, args, device, loss_fn, optimizer=None)
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            log_rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "loss": metrics["loss"],
                    "nrmse": metrics["nrmse"],
                    "psnr_db": metrics["psnr_db"],
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "grad_norm": metrics["grad_norm"],
                }
            )
        print(
            f"epoch {epoch}: train loss={train_metrics['loss']:.6f}, "
            f"train PSNR={train_metrics['psnr_db']:.3f}, "
            f"val loss={val_metrics['loss']:.6f}, val PSNR={val_metrics['psnr_db']:.3f}"
        )

        if val_metrics["psnr_db"] > best_val:
            best_val = val_metrics["psnr_db"]
            torch.save(model.state_dict(), save_dir / "best.pth")
        if args.saveEvery > 0 and epoch % args.saveEvery == 0:
            torch.save(model.state_dict(), save_dir / f"epoch{epoch}.pth")

    torch.save(model.state_dict(), save_dir / f"epoch{args.epochs}END.pth")
    write_log(save_dir / "training_log.csv", log_rows)

    # Inspect one validation batch after training, including fixed-point residuals.
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        y_reduced, x0, target = prepare_batch(batch, system, args, device, train=False)
        if args.solver == "finite":
            pred, state, history = model(y_reduced, x0=x0, return_state=True, return_history=True)
        else:
            pred, state = model(y_reduced, x0=x0, return_state=True)
            residuals = fixed_point.residuals(state, y_reduced)
            history = [
                {
                    "iter": "deq_final",
                    "fixed_point_abs_mean": float(residuals["fixed_point_abs"].mean().detach().cpu()),
                    "fixed_point_rel_mean": float(residuals["fixed_point_rel"].mean().detach().cpu()),
                    "primal_z_mean": float(residuals["primal_z"].mean().detach().cpu()),
                    "primal_s_mean": float(residuals["primal_s"].mean().detach().cpu()),
                    "data_residual_mean": float(residuals["data_residual"].mean().detach().cpu()),
                }
            ]
        final_smoke = {
            "finite_output": bool(torch.isfinite(pred).all().detach().cpu()),
            "shape_ok": tuple(pred.shape) == tuple(target.shape),
            "nonnegative_ok": bool(args.allowNegative or pred.min().detach().cpu() >= -1e-7),
            "loss_finite": bool(np.isfinite(log_rows[-1]["loss"])),
            "grad_nonzero": bool(log_rows[0]["grad_norm"] is not None and log_rows[0]["grad_norm"] > 0),
        }
        final_smoke["passed"] = all(final_smoke.values())

    summary = {
        "method": "Sparse-PnP-ADMM-DEQ",
        "solver": args.solver,
        "deqSolver": args.deqSolver if args.solver == "deq" else None,
        "denoiser": args.denoiser,
        "preloadDenoiser": args.preloadDenoiser if args.denoiser == "rdn" else None,
        "trainable_parameters": int(sum(param.numel() for param in trainable_params)),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - start_time,
        "best_val_psnr_db": best_val,
        "last_train": log_rows[-2],
        "last_val": log_rows[-1],
        "fixed_point_probe": history,
        "smoke_acceptance": final_smoke,
        "outputs": {
            "args_json": str(save_dir / "args.json"),
            "training_log_csv": str(save_dir / "training_log.csv"),
            "summary_json": str(save_dir / "summary.json"),
            "best_checkpoint": str(save_dir / "best.pth"),
            "final_checkpoint": str(save_dir / f"epoch{args.epochs}END.pth"),
        },
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["smoke_acceptance"], indent=2))

    if args.strictSmoke and not final_smoke["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
