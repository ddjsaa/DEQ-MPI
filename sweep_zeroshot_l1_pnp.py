import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_list(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run a compact ZeroShot-l1-PnP parameter sweep.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--denoiser", choices=["gaussian", "drunet"], default="drunet")
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")
    parser.add_argument("--testLimit", type=int, default=512)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--mu0s", default="1e4,1e5,2e5,5e5,1e6")
    parser.add_argument("--nIters", default="3,5,7,10")
    parser.add_argument("--alphaScales", default="0,0.001,0.005,0.01")
    parser.add_argument("--outDir", default="training/reproduction/zeroshot_sweep")
    parser.add_argument("--csv", default="training/reproduction/zeroshot_sweep_summary.csv")
    args = parser.parse_args()

    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mu0 in parse_list(args.mu0s, float):
        for n_iter in parse_list(args.nIters, int):
            for alpha_scale in parse_list(args.alphaScales, float):
                alpha = mu0 * alpha_scale
                tag = f"mu{mu0:g}_it{n_iter}_a{alpha_scale:g}".replace("+", "")
                out_json = out_dir / f"{tag}.json"
                cmd = [
                    args.python,
                    "eval_zeroshot_l1_pnp.py",
                    "--denoiser",
                    args.denoiser,
                    "--testLimit",
                    str(args.testLimit),
                    "--batchSize",
                    str(args.batchSize),
                    "--seed",
                    str(args.seed),
                    "--pSNRval",
                    str(args.pSNRval),
                    "--nIter",
                    str(n_iter),
                    "--mu0",
                    str(mu0),
                    "--alpha",
                    str(alpha),
                    "--outJson",
                    str(out_json),
                ]
                if args.denoiser == "drunet":
                    cmd.extend(["--dpirRoot", args.dpirRoot, "--drunetWeights", args.drunetWeights])

                print("Running", tag)
                subprocess.run(cmd, check=True)
                with open(out_json, "r", encoding="utf-8") as f:
                    result = json.load(f)
                rows.append(
                    {
                        "tag": tag,
                        "denoiser": args.denoiser,
                        "testLimit": result["num_test_images"],
                        "seed": result["seed"],
                        "mu0": result["mu0"],
                        "nIter": result["nIter"],
                        "alpha": result["alpha"],
                        "alphaScale": alpha_scale,
                        "mean_psnr_db": result["mean_psnr_db"],
                        "std_psnr_db": result["std_psnr_db"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "json": str(out_json),
                    }
                )

                rows_sorted = sorted(rows, key=lambda row: row["mean_psnr_db"], reverse=True)
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows_sorted)
                print(
                    "Best so far:",
                    rows_sorted[0]["tag"],
                    f"{rows_sorted[0]['mean_psnr_db']:.3f} dB",
                )

    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
