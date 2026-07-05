"""Summarize the minimal Sparse-PnP-DEQ ablation results.

Example:
  python make_sparse_pnp_deq_ablation_summary.py

The script consumes paired-evaluation JSON artifacts and writes compact CSV,
JSON, and Markdown tables. It does not run model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUN_DIR = Path(
    "training/reproduction/sparse_pnp_admm_deq/"
    "deq_anderson_rdn_full_e3_it12_eta0p10"
)
DEFAULT_PAIRED_DIR = DEFAULT_RUN_DIR / "paired_evaluations"
DEFAULT_ABLATION_DIR = DEFAULT_RUN_DIR / "ablations"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Sparse-PnP-DEQ ablations")
    parser.add_argument("--pairedDir", default=str(DEFAULT_PAIRED_DIR))
    parser.add_argument("--ablationDir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--outDir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--prefix", default="sparse_pnp_deq_minimal_ablation")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict, name: str, stat: str = "mean"):
    return float(summary["metrics"][name][stat])


def timing(summary: dict, name: str):
    return float(summary["timing"][name])


def speedup(summary: dict):
    sparse = timing(summary, "sparse_elapsed_seconds")
    deq = timing(summary, "deq_mpi_elapsed_seconds")
    return deq / sparse if sparse > 0 else None


def format_float(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def default_items(paired_dir: Path, ablation_dir: Path):
    return [
        {
            "group": "l1",
            "label": "current_l1_quality",
            "path": paired_dir / "paired_canonical_full_seed2026_summary.json",
            "solver": "deq",
            "deq_solver": "anderson",
            "max_iter": 12,
            "eta": 0.10,
            "l1_lambda": 0.0005,
            "denoiser": "trained_rdn",
            "note": "Quality baseline for single-seed ablations.",
        },
        {
            "group": "l1",
            "label": "no_l1",
            "path": ablation_dir / "ablation_canonical_full_seed2026_no_l1_summary.json",
            "solver": "deq",
            "deq_solver": "anderson",
            "max_iter": 12,
            "eta": 0.10,
            "l1_lambda": 0.0,
            "denoiser": "trained_rdn",
            "note": "Inference-time l1 removed.",
        },
        {
            "group": "l1",
            "label": "tiny_l1",
            "path": ablation_dir / "ablation_canonical_full_seed2026_l1p0002_summary.json",
            "solver": "deq",
            "deq_solver": "anderson",
            "max_iter": 12,
            "eta": 0.10,
            "l1_lambda": 0.0002,
            "denoiser": "trained_rdn",
            "note": "Inference-time l1 reduced.",
        },
        {
            "group": "denoiser",
            "label": "pretrained_rdn_no_sparse_training",
            "path": ablation_dir
            / "ablation_canonical_full_seed2026_pretrained_rdn_skipckpt_summary.json",
            "solver": "deq",
            "deq_solver": "anderson",
            "max_iter": 12,
            "eta": 0.10,
            "l1_lambda": 0.0005,
            "denoiser": "pretrained_rdn",
            "note": "Sparse checkpoint skipped; denoiser is only preloaded.",
        },
        {
            "group": "denoiser",
            "label": "identity_no_denoiser",
            "path": ablation_dir / "ablation_canonical_full_seed2026_identity_skipckpt_summary.json",
            "solver": "deq",
            "deq_solver": "anderson",
            "max_iter": 12,
            "eta": 0.10,
            "l1_lambda": 0.0005,
            "denoiser": "identity",
            "note": "Denoiser branch replaced by identity.",
        },
        {
            "group": "solver",
            "label": "finite_it12",
            "path": ablation_dir / "ablation_canonical_full_seed2026_finite_it12_summary.json",
            "solver": "finite",
            "deq_solver": "",
            "max_iter": 12,
            "eta": 0.10,
            "l1_lambda": 0.0005,
            "denoiser": "trained_rdn",
            "note": "Same denoiser checkpoint, finite fixed-point iterations.",
        },
        {
            "group": "speed",
            "label": "fast_deq_it10_eta0p001",
            "path": paired_dir / "paired_canonical_full_seed2026_it10_eta0p001_summary.json",
            "solver": "deq",
            "deq_solver": "anderson",
            "max_iter": 10,
            "eta": 0.001,
            "l1_lambda": 0.0005,
            "denoiser": "trained_rdn",
            "note": "Fast inference setting.",
        },
    ]


def build_rows(items):
    rows = []
    missing = []
    for item in items:
        path = Path(item["path"])
        if not path.exists():
            missing.append(str(path))
            continue
        summary = read_json(path)
        sparse_psnr = metric(summary, "sparse_psnr_db")
        deq_psnr = metric(summary, "deq_mpi_psnr_db")
        delta = metric(summary, "delta_sparse_minus_deq_mpi_db")
        row = {
            "group": item["group"],
            "label": item["label"],
            "seed": summary["seed"],
            "num_samples": summary["num_samples"],
            "solver": item["solver"],
            "deq_solver": item["deq_solver"],
            "max_iter": item["max_iter"],
            "eta": item["eta"],
            "l1_lambda": item["l1_lambda"],
            "denoiser": item["denoiser"],
            "sparse_psnr_db": sparse_psnr,
            "deq_mpi_psnr_db": deq_psnr,
            "delta_sparse_minus_deq_mpi_db": delta,
            "sparse_win_rate": float(summary["win_rates"]["delta_gt_0"]),
            "speedup_deq_over_sparse": speedup(summary),
            "sparse_elapsed_seconds": timing(summary, "sparse_elapsed_seconds"),
            "deq_mpi_elapsed_seconds": timing(summary, "deq_mpi_elapsed_seconds"),
            "sparse_checkpoint": summary.get("sparse_checkpoint"),
            "source_summary_json": str(path),
            "note": item["note"],
        }
        rows.append(row)
    if missing:
        raise FileNotFoundError("Missing ablation summaries:\n" + "\n".join(missing))

    baseline = next(row for row in rows if row["label"] == "current_l1_quality")
    baseline_psnr = baseline["sparse_psnr_db"]
    baseline_delta = baseline["delta_sparse_minus_deq_mpi_db"]
    for row in rows:
        row["delta_psnr_vs_current_l1_db"] = row["sparse_psnr_db"] - baseline_psnr
        row["delta_delta_vs_current_l1_db"] = (
            row["delta_sparse_minus_deq_mpi_db"] - baseline_delta
        )
    return rows


def key_findings(rows):
    by_label = {row["label"]: row for row in rows}
    current = by_label["current_l1_quality"]
    return {
        "seed": current["seed"],
        "num_samples": current["num_samples"],
        "current_l1_delta_vs_deq_mpi_db": current["delta_sparse_minus_deq_mpi_db"],
        "no_l1_minus_current_psnr_db": by_label["no_l1"]["delta_psnr_vs_current_l1_db"],
        "tiny_l1_minus_current_psnr_db": by_label["tiny_l1"]["delta_psnr_vs_current_l1_db"],
        "pretrained_rdn_drop_vs_current_db": -by_label[
            "pretrained_rdn_no_sparse_training"
        ]["delta_psnr_vs_current_l1_db"],
        "identity_no_denoiser_drop_vs_current_db": -by_label["identity_no_denoiser"][
            "delta_psnr_vs_current_l1_db"
        ],
        "finite_it12_minus_deq_it12_psnr_db": by_label["finite_it12"][
            "delta_psnr_vs_current_l1_db"
        ],
        "fast_it10_drop_vs_current_db": -by_label["fast_deq_it10_eta0p001"][
            "delta_psnr_vs_current_l1_db"
        ],
        "fast_it10_speedup": by_label["fast_deq_it10_eta0p001"][
            "speedup_deq_over_sparse"
        ],
    }


def write_csv(path: Path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows, findings):
    lines = [
        "# Sparse-PnP-DEQ Minimal Ablation",
        "",
        (
            f"- Scope: canonical full-test paired evaluation, seed `{findings['seed']}`, "
            f"samples `{findings['num_samples']}`."
        ),
        (
            "- Note: this table is single-seed ablation evidence; the main quality claim "
            "still uses the existing multi-seed paired summary."
        ),
        "",
        "## Key Findings",
        (
            f"- Current l1 quality delta vs DEQ-MPI: "
            f"`{findings['current_l1_delta_vs_deq_mpi_db']:.4f} dB`."
        ),
        (
            f"- l1 strength is weakly influential here: no-l1/current "
            f"`{findings['no_l1_minus_current_psnr_db']:+.4f} dB`, tiny-l1/current "
            f"`{findings['tiny_l1_minus_current_psnr_db']:+.4f} dB`."
        ),
        (
            f"- Sparse supervised tuning matters: pretrained RDN drops "
            f"`{findings['pretrained_rdn_drop_vs_current_db']:.4f} dB`."
        ),
        (
            f"- The denoiser branch is essential: identity/no-denoiser drops "
            f"`{findings['identity_no_denoiser_drop_vs_current_db']:.4f} dB`."
        ),
        (
            f"- Finite it12 is `{findings['finite_it12_minus_deq_it12_psnr_db']:+.4f} dB` "
            "relative to DEQ-Anderson it12 with the same denoiser checkpoint."
        ),
        (
            f"- Fast it10 eta=0.001 drops `{findings['fast_it10_drop_vs_current_db']:.4f} dB` "
            f"and runs at `{findings['fast_it10_speedup']:.2f}x` DEQ-MPI wall-time speedup."
        ),
        "",
        "## Table",
        "",
        "| Group | Setting | Sparse PSNR | Delta vs DEQ-MPI | Delta vs current | Speedup | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['group']} | "
            f"{row['label']} | "
            f"{format_float(row['sparse_psnr_db'])} | "
            f"{format_float(row['delta_sparse_minus_deq_mpi_db'])} | "
            f"{format_float(row['delta_psnr_vs_current_l1_db'], digits=4)} | "
            f"{format_float(row['speedup_deq_over_sparse'], digits=2)}x | "
            f"{row['note']} |"
        )
    lines.extend(["", "## Sources"])
    for row in rows:
        lines.append(f"- `{row['label']}`: `{row['source_summary_json']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = default_items(Path(args.pairedDir), Path(args.ablationDir))
    rows = build_rows(items)
    findings = key_findings(rows)

    csv_path = out_dir / f"{args.prefix}.csv"
    json_path = out_dir / f"{args.prefix}.json"
    md_path = out_dir / f"{args.prefix}.md"
    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "key_findings": findings}, f, indent=2)
    write_markdown(md_path, rows, findings)

    print(
        json.dumps(
            {
                "key_findings": findings,
                "outputs": {
                    "csv": str(csv_path),
                    "json": str(json_path),
                    "markdown": str(md_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
