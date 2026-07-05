import argparse
import csv
import json
from pathlib import Path


def load_row(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "mean_deq_psnr_db" in data:
        method = "DEQ-MPI"
        mean = data["mean_deq_psnr_db"]
        std = data["std_deq_psnr_db"]
        min_val = data["min_deq_psnr_db"]
        max_val = data["max_deq_psnr_db"]
        config = f"maxIter={data.get('maxIter')}, svd={data.get('nbOfSingulars')}"
    else:
        method = data.get("method", path.stem)
        if data.get("alpha", None) == 0:
            method = method.replace("l1-", "")
        mean = data["mean_psnr_db"]
        std = data["std_psnr_db"]
        min_val = data["min_psnr_db"]
        max_val = data["max_psnr_db"]
        config = (
            f"denoiser={data.get('denoiser')}, nIter={data.get('nIter')}, "
            f"mu0={data.get('mu0')}, alpha={data.get('alpha')}, svd={data.get('nbOfSingulars')}"
        )
    return {
        "method": method,
        "num_test_images": data.get("num_test_images"),
        "seed": data.get("seed", ""),
        "pSNRval": data.get("pSNRval"),
        "mean_psnr_db": mean,
        "std_psnr_db": std,
        "min_psnr_db": min_val,
        "max_psnr_db": max_val,
        "elapsed_seconds": data.get("elapsed_seconds"),
        "config": config,
        "json": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize DEQ-MPI and ZeroShot-PnP JSON metrics.")
    parser.add_argument("jsons", nargs="+")
    parser.add_argument("--csv", default="training/reproduction/repro_comparison_summary.csv")
    parser.add_argument("--md", default="training/reproduction/repro_comparison_summary.md")
    args = parser.parse_args()

    rows = [load_row(Path(p)) for p in args.jsons]
    rows = sorted(rows, key=lambda row: row["mean_psnr_db"], reverse=True)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | N | Mean PSNR (dB) | Std | Min | Max | Time (s) | Config |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['num_test_images']} | {row['mean_psnr_db']:.3f} | "
                f"{row['std_psnr_db']:.3f} | {row['min_psnr_db']:.3f} | {row['max_psnr_db']:.3f} | "
                f"{row['elapsed_seconds']:.3f} | {row['config']} |\n"
            )

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    print("Best:", rows[0]["method"], f"{rows[0]['mean_psnr_db']:.3f} dB")


if __name__ == "__main__":
    main()
