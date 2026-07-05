"""Summarize Sparse-PnP-DEQ stress-test paired evaluations.

Example:
  python make_sparse_pnp_deq_stress_summary.py

The script consumes paired-evaluation JSON artifacts and writes CSV/JSON/MD
summaries plus a compact stress-test figure. It does not run inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_DIR = Path(
    "training/reproduction/sparse_pnp_admm_deq/"
    "deq_anderson_rdn_full_e3_it12_eta0p10"
)
DEFAULT_PAIRED_DIR = DEFAULT_RUN_DIR / "paired_evaluations"
DEFAULT_STRESS_DIR = DEFAULT_RUN_DIR / "stress_tests"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Sparse-PnP-DEQ stress tests")
    parser.add_argument("--pairedDir", default=str(DEFAULT_PAIRED_DIR))
    parser.add_argument("--stressDir", default=str(DEFAULT_STRESS_DIR))
    parser.add_argument("--outDir", default=str(DEFAULT_STRESS_DIR))
    parser.add_argument("--prefix", default="sparse_pnp_deq_stress_tests")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict, name: str, stat: str = "mean"):
    return float(summary["metrics"][name][stat])


def speedup(summary: dict):
    sparse = float(summary["timing"]["sparse_elapsed_seconds"])
    deq = float(summary["timing"]["deq_mpi_elapsed_seconds"])
    return deq / sparse if sparse > 0 else None


def item_specs(paired_dir: Path, stress_dir: Path):
    return [
        {
            "axis": "pSNR",
            "label": "pSNR8",
            "value": 8.0,
            "path": stress_dir / "stress_canonical_full_seed2026_pSNR8_summary.json",
            "note": "Lower SNR than training setting.",
        },
        {
            "axis": "pSNR",
            "label": "pSNR10",
            "value": 10.0,
            "path": paired_dir / "paired_canonical_full_seed2026_summary.json",
            "note": "Training-noise canonical baseline.",
        },
        {
            "axis": "pSNR",
            "label": "pSNR12",
            "value": 12.0,
            "path": stress_dir / "stress_canonical_full_seed2026_pSNR12_summary.json",
            "note": "Moderately cleaner measurements.",
        },
        {
            "axis": "pSNR",
            "label": "pSNR15",
            "value": 15.0,
            "path": stress_dir / "stress_canonical_full_seed2026_pSNR15_summary.json",
            "note": "Cleaner measurements.",
        },
        {
            "axis": "rank",
            "label": "rank180",
            "value": 180.0,
            "path": stress_dir / "stress_canonical_full_seed2026_rank180_summary.json",
            "note": "Lower SVD truncation rank.",
        },
        {
            "axis": "rank",
            "label": "rank220",
            "value": 220.0,
            "path": paired_dir / "paired_canonical_full_seed2026_summary.json",
            "note": "Training-rank canonical baseline.",
        },
        {
            "axis": "rank",
            "label": "rank250",
            "value": 250.0,
            "path": stress_dir / "stress_canonical_full_seed2026_rank250_summary.json",
            "note": "Higher SVD truncation rank.",
        },
        {
            "axis": "mode",
            "label": "direct",
            "value": 0.0,
            "path": paired_dir / "paired_direct_full_seed2026_summary.json",
            "note": "Direct reconstruction/measurement system.",
        },
        {
            "axis": "mode",
            "label": "canonical",
            "value": 1.0,
            "path": paired_dir / "paired_canonical_full_seed2026_summary.json",
            "note": "Canonical no-inverse-crime system.",
        },
    ]


def build_rows(specs):
    rows = []
    missing = []
    for spec in specs:
        path = Path(spec["path"])
        if not path.exists():
            missing.append(str(path))
            continue
        summary = read_json(path)
        rows.append(
            {
                "axis": spec["axis"],
                "label": spec["label"],
                "value": spec["value"],
                "mode": summary["mode"],
                "seed": summary["seed"],
                "num_samples": summary["num_samples"],
                "pSNRval": float(summary["problem"]["pSNRval"]),
                "measured_snr_db": float(summary["problem"]["measured_snr_db"]),
                "nbOfSingulars": int(summary["problem"]["nbOfSingulars"]),
                "sparse_psnr_db": metric(summary, "sparse_psnr_db"),
                "deq_mpi_psnr_db": metric(summary, "deq_mpi_psnr_db"),
                "delta_sparse_minus_deq_mpi_db": metric(
                    summary, "delta_sparse_minus_deq_mpi_db"
                ),
                "sparse_win_rate": float(summary["win_rates"]["delta_gt_0"]),
                "speedup_deq_over_sparse": speedup(summary),
                "sparse_data_residual_norm": metric(summary, "sparse_data_residual_norm"),
                "deq_mpi_data_residual_norm": metric(summary, "deq_mpi_data_residual_norm"),
                "source_summary_json": str(path),
                "note": spec["note"],
            }
        )
    if missing:
        raise FileNotFoundError("Missing stress summaries:\n" + "\n".join(missing))
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def key_findings(rows):
    psnr_rows = sorted([row for row in rows if row["axis"] == "pSNR"], key=lambda r: r["value"])
    rank_rows = sorted([row for row in rows if row["axis"] == "rank"], key=lambda r: r["value"])
    mode_rows = {row["label"]: row for row in rows if row["axis"] == "mode"}
    psnr_deltas = [row["delta_sparse_minus_deq_mpi_db"] for row in psnr_rows]
    rank_deltas = [row["delta_sparse_minus_deq_mpi_db"] for row in rank_rows]
    return {
        "seed": psnr_rows[0]["seed"],
        "num_samples": psnr_rows[0]["num_samples"],
        "pSNR_delta_min_db": min(psnr_deltas),
        "pSNR_delta_max_db": max(psnr_deltas),
        "pSNR_delta_range_db": max(psnr_deltas) - min(psnr_deltas),
        "rank_delta_min_db": min(rank_deltas),
        "rank_delta_max_db": max(rank_deltas),
        "rank_delta_range_db": max(rank_deltas) - min(rank_deltas),
        "direct_delta_db": mode_rows["direct"]["delta_sparse_minus_deq_mpi_db"],
        "canonical_delta_db": mode_rows["canonical"]["delta_sparse_minus_deq_mpi_db"],
        "direct_minus_canonical_delta_db": mode_rows["direct"][
            "delta_sparse_minus_deq_mpi_db"
        ]
        - mode_rows["canonical"]["delta_sparse_minus_deq_mpi_db"],
        "mean_speedup": float(np.mean([row["speedup_deq_over_sparse"] for row in rows])),
    }


def style_axes(ax):
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(length=3, width=0.7, color="#555555")


def panel_label(ax, label):
    ax.text(
        -0.17,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def save_figure(rows, path_base: Path):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    blue = "#4e79a7"
    teal = "#59a89c"
    amber = "#f28e2b"
    gray = "#a7adb5"
    neutral = "#505a66"

    fig = plt.figure(figsize=(7.1, 2.55))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.98, bottom=0.22, top=0.82, wspace=0.36)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]

    psnr_rows = sorted([row for row in rows if row["axis"] == "pSNR"], key=lambda r: r["value"])
    x = [row["value"] for row in psnr_rows]
    y = [row["delta_sparse_minus_deq_mpi_db"] for row in psnr_rows]
    axes[0].axhline(0.0, color=gray, linestyle="--", linewidth=0.8)
    axes[0].axhline(-0.1, color="#c7ccd1", linestyle=":", linewidth=0.8)
    axes[0].plot(x, y, color=blue, marker="o", linewidth=1.8)
    axes[0].set_xlabel("pSNR setting (dB)")
    axes[0].set_ylabel("Delta PSNR vs DEQ-MPI (dB)")
    axes[0].set_title("Noise stress", loc="left", fontsize=8)
    axes[0].set_xticks(x)
    style_axes(axes[0])
    panel_label(axes[0], "A")

    rank_rows = sorted([row for row in rows if row["axis"] == "rank"], key=lambda r: r["value"])
    x = [row["value"] for row in rank_rows]
    y = [row["delta_sparse_minus_deq_mpi_db"] for row in rank_rows]
    axes[1].axhline(0.0, color=gray, linestyle="--", linewidth=0.8)
    axes[1].axhline(-0.1, color="#c7ccd1", linestyle=":", linewidth=0.8)
    axes[1].plot(x, y, color=teal, marker="o", linewidth=1.8)
    axes[1].set_xlabel("SVD rank")
    axes[1].set_ylabel("Delta PSNR vs DEQ-MPI (dB)")
    axes[1].set_title("Rank stress", loc="left", fontsize=8)
    axes[1].set_xticks(x)
    style_axes(axes[1])
    panel_label(axes[1], "B")

    mode_rows = [row for row in rows if row["axis"] == "mode"]
    labels = [row["label"].capitalize() for row in mode_rows]
    y = [row["delta_sparse_minus_deq_mpi_db"] for row in mode_rows]
    colors = [amber if row["label"] == "direct" else neutral for row in mode_rows]
    axes[2].axhline(0.0, color=gray, linestyle="--", linewidth=0.8)
    axes[2].bar(np.arange(len(y)), y, color=colors, alpha=0.78, width=0.58)
    axes[2].set_xticks(np.arange(len(y)))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Delta PSNR vs DEQ-MPI (dB)")
    axes[2].set_title("System mismatch", loc="left", fontsize=8)
    style_axes(axes[2])
    panel_label(axes[2], "C")

    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, rows, findings):
    lines = [
        "# Sparse-PnP-DEQ Stress Tests",
        "",
        (
            f"- Scope: full-test paired evaluation, seed `{findings['seed']}`, "
            f"samples `{findings['num_samples']}`."
        ),
        "- Method: quality Sparse-PnP-DEQ setting unless otherwise noted.",
        "",
        "## Key Findings",
        (
            f"- Across pSNR 8/10/12/15, delta ranges from "
            f"`{findings['pSNR_delta_min_db']:.4f}` to "
            f"`{findings['pSNR_delta_max_db']:.4f} dB`."
        ),
        (
            f"- Across SVD ranks 180/220/250, delta stays within "
            f"`{findings['rank_delta_min_db']:.4f}` to "
            f"`{findings['rank_delta_max_db']:.4f} dB`."
        ),
        (
            f"- Direct mode delta is `{findings['direct_delta_db']:.4f} dB`; "
            f"canonical mode delta is `{findings['canonical_delta_db']:.4f} dB`."
        ),
        f"- Mean wall-time speedup over DEQ-MPI across listed stress points: `{findings['mean_speedup']:.2f}x`.",
        "",
        "## Table",
        "",
        "| Axis | Setting | Sparse PSNR | DEQ-MPI PSNR | Delta | Win rate | Speedup | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['axis']} | "
            f"{row['label']} | "
            f"{f(row['sparse_psnr_db'])} | "
            f"{f(row['deq_mpi_psnr_db'])} | "
            f"{f(row['delta_sparse_minus_deq_mpi_db'])} | "
            f"{100 * row['sparse_win_rate']:.2f}% | "
            f"{f(row['speedup_deq_over_sparse'], 2)}x | "
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
    rows = build_rows(item_specs(Path(args.pairedDir), Path(args.stressDir)))
    findings = key_findings(rows)

    csv_path = out_dir / f"{args.prefix}.csv"
    json_path = out_dir / f"{args.prefix}.json"
    md_path = out_dir / f"{args.prefix}.md"
    figure_base = out_dir / args.prefix

    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "key_findings": findings}, f, indent=2)
    write_markdown(md_path, rows, findings)
    save_figure(rows, figure_base)

    print(
        json.dumps(
            {
                "key_findings": findings,
                "outputs": {
                    "csv": str(csv_path),
                    "json": str(json_path),
                    "markdown": str(md_path),
                    "png": str(figure_base.with_suffix(".png")),
                    "pdf": str(figure_base.with_suffix(".pdf")),
                    "svg": str(figure_base.with_suffix(".svg")),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
