import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_csv_values(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def summarize_recon(metrics_path: Path, recon_path: Path, scale: str):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recon = np.load(recon_path)["recon"]
    q = np.quantile(recon, [0.5, 0.95, 0.99])
    return {
        "phantom": metrics_path.name.split(f"_{scale}_", 1)[0].replace("openmpi_", ""),
        "scale": scale,
        "rows": metrics["A_shape"][0],
        "denoiser": metrics["denoiser"],
        "alpha": metrics["alpha"],
        "mu0": metrics["mu0"],
        "nIter": metrics["nIter"],
        "mean": float(recon.mean()),
        "max": float(recon.max()),
        "median": float(q[0]),
        "p95": float(q[1]),
        "p99": float(q[2]),
        "nonzero_gt_1e-4": float((recon > 1e-4).mean()),
        "final_residual": metrics["history"][-1]["data_residual"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "metrics": str(metrics_path),
        "recon": str(recon_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep OpenMPI 3D ZeroShot alpha values.")
    parser.add_argument("--phantoms", default="resolution,concentration")
    parser.add_argument("--alphas", default="1000,2500,5000,10000")
    parser.add_argument("--denoiser", choices=["gaussian", "drunet"], default="drunet")
    parser.add_argument("--scale", default="medium", help="NPZ/result scale label, e.g. medium or large.")
    parser.add_argument("--npzRoot", default="datasets/OpenMPI")
    parser.add_argument("--outRoot", default="training/reproduction")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nIter", type=int, default=5)
    parser.add_argument("--mu0", type=float, default=5e5)
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")
    parser.add_argument("--summaryCsv", default=None)
    args = parser.parse_args()

    phantoms = parse_csv_values(args.phantoms, str)
    alphas = parse_csv_values(args.alphas, float)
    out_root = Path(args.outRoot)
    rows = []

    for phantom in phantoms:
        npz = Path(args.npzRoot) / f"{phantom}_{args.scale}_real_system.npz"
        if not npz.exists():
            raise FileNotFoundError(npz)
        for alpha in alphas:
            alpha_label = str(int(alpha)) if float(alpha).is_integer() else str(alpha).replace(".", "p")
            stem = f"openmpi_{phantom}_{args.scale}_zeroshot_l1_pnp_{args.denoiser}_alpha{alpha_label}"
            recon_path = out_root / f"{stem}_recon.npz"
            metrics_path = out_root / f"{stem}_metrics.json"
            cmd = [
                sys.executable,
                "eval_zeroshot_openmpi3d.py",
                "--npz",
                str(npz),
                "--gpu",
                str(args.gpu),
                "--nIter",
                str(args.nIter),
                "--mu0",
                str(args.mu0),
                "--alpha",
                str(alpha),
                "--cg",
                "--denoiser",
                args.denoiser,
                "--outRecon",
                str(recon_path),
                "--outJson",
                str(metrics_path),
            ]
            if args.denoiser == "drunet":
                cmd.extend(["--dpirRoot", args.dpirRoot, "--drunetWeights", args.drunetWeights])
            print("running", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            rows.append(summarize_recon(metrics_path, recon_path, args.scale))

    summary_csv = Path(args.summaryCsv or out_root / f"openmpi_{args.scale}_{args.denoiser}_alpha_sweep.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(summary_csv)


if __name__ == "__main__":
    main()
