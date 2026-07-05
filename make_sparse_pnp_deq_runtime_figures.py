"""Create paired Sparse-PnP-DEQ quality/runtime summary figures.

Example:
  python make_sparse_pnp_deq_runtime_figures.py

The script consumes paired evaluation CSV/JSON artifacts and writes a
publication-style multi-panel figure plus source-data tables. It does not run
model inference.
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
DEFAULT_EVAL_DIR = DEFAULT_RUN_DIR / "paired_evaluations"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make Sparse-PnP-DEQ paired quality/runtime summary figures."
    )
    parser.add_argument("--evalDir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--outDir", default=str(DEFAULT_RUN_DIR / "figures"))
    parser.add_argument("--prefix", default="sparse_pnp_deq_quality_runtime")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def f(row, key):
    return float(row[key])


def panel_label(ax, label):
    ax.text(
        -0.14,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def style_axes(ax):
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.7)
    ax.tick_params(axis="both", length=3, width=0.7, color="#555555")
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def save_figure(fig, base_path: Path):
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def write_source_data(path: Path, maxiter_rows, eta_rows, multiseed_rows, trace_rows):
    rows = []
    for row in maxiter_rows:
        rows.extend(
            [
                {
                    "panel": "A",
                    "series": "delta_sparse_minus_deq_mpi_db",
                    "setting": f"maxIter={row['sparse_max_iter']}",
                    "x": row["sparse_max_iter"],
                    "y": row["delta_sparse_minus_deq_mpi_db"],
                },
                {
                    "panel": "A",
                    "series": "speedup_deq_over_sparse",
                    "setting": f"maxIter={row['sparse_max_iter']}",
                    "x": row["sparse_max_iter"],
                    "y": row["speedup_deq_over_sparse"],
                },
            ]
        )
    for row in eta_rows:
        rows.append(
            {
                "panel": "B",
                "series": "delta_sparse_minus_deq_mpi_db",
                "setting": f"eta={row['sparse_eta']}",
                "x": row["sparse_eta"],
                "y": row["delta_sparse_minus_deq_mpi_db"],
            }
        )
    for row in multiseed_rows:
        rows.extend(
            [
                {
                    "panel": "C",
                    "series": "quality_delta_sparse_minus_deq_mpi_db",
                    "setting": f"seed={row['seed']}",
                    "x": "quality_it12_eta0p10",
                    "y": row["quality_it12_delta_sparse_minus_deq_mpi_db"],
                },
                {
                    "panel": "C",
                    "series": "fast_delta_sparse_minus_deq_mpi_db",
                    "setting": f"seed={row['seed']}",
                    "x": "fast_it10_eta0p001",
                    "y": row["fast_delta_sparse_minus_deq_mpi_db"],
                },
                {
                    "panel": "C",
                    "series": "fast_speedup_deq_over_sparse",
                    "setting": f"seed={row['seed']}",
                    "x": "fast_it10_eta0p001",
                    "y": row["fast_speedup_deq_over_sparse"],
                },
            ]
        )
    for row in trace_rows:
        rows.extend(
            [
                {
                    "panel": "D",
                    "series": "mean_solver_residual",
                    "setting": row["setting"],
                    "x": row["solver_iter"],
                    "y": row["mean_solver_residual"],
                },
                {
                    "panel": "D",
                    "series": "q10_solver_residual",
                    "setting": row["setting"],
                    "x": row["solver_iter"],
                    "y": row["q10_solver_residual"],
                },
                {
                    "panel": "D",
                    "series": "q90_solver_residual",
                    "setting": row["setting"],
                    "x": row["solver_iter"],
                    "y": row["q90_solver_residual"],
                },
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["panel", "series", "setting", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)


def make_figure(eval_dir: Path, out_dir: Path, prefix: str):
    maxiter_rows = read_csv(eval_dir / "paired_canonical_seed2026_maxiter_sweep.csv")
    eta_rows = read_csv(eval_dir / "paired_canonical_seed2026_it10_eta_sweep.csv")
    multiseed_rows = read_csv(eval_dir / "paired_canonical_it10_eta0p001_multiseed_summary.csv")
    paired_summary = read_json(eval_dir / "paired_evaluation_summary.json")
    fast_summary = read_json(eval_dir / "paired_canonical_it10_eta0p001_multiseed_summary.json")
    eta_summary = read_json(eval_dir / "paired_canonical_seed2026_it10_eta_sweep.json")
    trace_dir = eval_dir.parent / "convergence_traces"
    trace_rows = read_csv(trace_dir / "convergence_canonical_seed2026_full_solver_trace.csv")
    trace_summary = read_json(trace_dir / "convergence_canonical_seed2026_full_summary.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_csv = out_dir / f"{prefix}_source_data.csv"
    write_source_data(source_csv, maxiter_rows, eta_rows, multiseed_rows, trace_rows)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    neutral = "#505a66"
    blue = "#4e79a7"
    teal = "#59a89c"
    amber = "#f28e2b"
    red = "#d65f5f"
    gray = "#a7adb5"

    fig = plt.figure(figsize=(7.1, 5.55))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.83,
        wspace=0.34,
        hspace=0.42,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A: quality/runtime frontier from maxIter sweep.
    iters = np.array([f(row, "sparse_max_iter") for row in maxiter_rows])
    delta = np.array([f(row, "delta_sparse_minus_deq_mpi_db") for row in maxiter_rows])
    speed = np.array([f(row, "speedup_deq_over_sparse") for row in maxiter_rows])
    ax_a.axhline(0.0, color=gray, linewidth=0.8, linestyle="--")
    ax_a.axhline(-0.1, color="#c7ccd1", linewidth=0.8, linestyle=":")
    ax_a.plot(iters, delta, marker="o", color=blue, linewidth=1.8, label="Delta PSNR")
    ax_a.scatter([10, 12], delta[[2, 3]], s=42, color=[amber, teal], zorder=4)
    ax_a.set_xlabel("Sparse-PnP maxIter")
    ax_a.set_ylabel("Delta PSNR vs DEQ-MPI (dB)", color=blue)
    ax_a.tick_params(axis="y", labelcolor=blue)
    ax_a.set_xticks(iters)
    ax_a.set_ylim(-1.62, 0.12)
    ax_a2 = ax_a.twinx()
    ax_a2.plot(iters, speed, marker="s", color=neutral, linewidth=1.4, label="Speedup")
    ax_a2.set_ylabel("Wall-time speedup", color=neutral)
    ax_a2.tick_params(axis="y", labelcolor=neutral)
    ax_a2.set_ylim(2.2, 5.2)
    ax_a2.spines["top"].set_visible(False)
    ax_a2.spines["right"].set_color("#666666")
    style_axes(ax_a)
    ax_a.text(11.45, delta[-1] + 0.055, "Delta PSNR", color=blue, fontsize=7, ha="right")
    ax_a2.text(9.1, speed[2] + 0.18, "Speedup", color=neutral, fontsize=7)
    panel_label(ax_a, "A")
    ax_a.set_title("Quality-speed frontier", loc="left", fontsize=8)

    # Panel B: eta sweep for the fast setting.
    eta_vals = np.array([f(row, "sparse_eta") for row in eta_rows])
    eta_delta = np.array([f(row, "delta_sparse_minus_deq_mpi_db") for row in eta_rows])
    x = np.arange(len(eta_vals))
    best_idx = int(np.argmax(eta_delta))
    colors = [teal if idx == best_idx else blue for idx in range(len(x))]
    ax_b.axhline(0.0, color=gray, linewidth=0.8, linestyle="--")
    ax_b.axhline(-0.1, color="#c7ccd1", linewidth=0.8, linestyle=":")
    ax_b.plot(x, eta_delta, color=blue, linewidth=1.4)
    ax_b.scatter(x, eta_delta, s=28, color=colors, zorder=3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(row["sparse_eta"]) for row in eta_rows], rotation=45, ha="right")
    ax_b.set_xlabel("eta at maxIter=10")
    ax_b.set_ylabel("Delta PSNR vs DEQ-MPI (dB)")
    ax_b.set_ylim(-0.125, -0.055)
    style_axes(ax_b)
    panel_label(ax_b, "B")
    ax_b.set_title("Fast setting tuning", loc="left", fontsize=8)
    ax_b.annotate(
        "best tested",
        xy=(best_idx, eta_delta[best_idx]),
        xytext=(best_idx + 0.8, eta_delta[best_idx] + 0.007),
        arrowprops={"arrowstyle": "-", "color": teal, "linewidth": 0.8},
        color=teal,
        fontsize=7,
    )

    # Panel C: multi-seed paired validation for quality and fast settings.
    seeds = [row["seed"] for row in multiseed_rows]
    quality_delta = np.array([f(row, "quality_it12_delta_sparse_minus_deq_mpi_db") for row in multiseed_rows])
    fast_delta = np.array([f(row, "fast_delta_sparse_minus_deq_mpi_db") for row in multiseed_rows])
    group_x = np.array([0, 1])
    ax_c.axhline(0.0, color=gray, linewidth=0.8, linestyle="--")
    means = [float(np.mean(quality_delta)), float(np.mean(fast_delta))]
    ax_c.bar(group_x, means, width=0.58, color=[teal, amber], alpha=0.78)
    jitter = np.array([-0.08, 0.0, 0.08])
    for idx, seed in enumerate(seeds):
        ax_c.plot(
            group_x + jitter[idx],
            [quality_delta[idx], fast_delta[idx]],
            color="#c5c9ce",
            linewidth=0.8,
            zorder=1,
        )
        ax_c.scatter(group_x + jitter[idx], [quality_delta[idx], fast_delta[idx]], s=22, color=neutral, zorder=3)
    ax_c.set_xticks(group_x)
    ax_c.set_xticklabels(["Quality\nit12 eta=0.10", "Fast\nit10 eta=0.001"])
    ax_c.set_ylabel("Delta PSNR vs DEQ-MPI (dB)")
    ax_c.set_ylim(-0.09, 0.025)
    style_axes(ax_c)
    panel_label(ax_c, "C")
    ax_c.set_title("Paired multi-seed validation", loc="left", fontsize=8)

    # Panel D: full-test Anderson residual trace.
    trace_by_setting = {
        "quality_it12_eta0p10": {
            "label": "Quality",
            "color": teal,
            "rows": [],
        },
        "fast_it10_eta0p001": {
            "label": "Fast",
            "color": amber,
            "rows": [],
        },
    }
    for row in trace_rows:
        if row["setting"] in trace_by_setting:
            trace_by_setting[row["setting"]]["rows"].append(row)
    for payload in trace_by_setting.values():
        rows = sorted(payload["rows"], key=lambda item: f(item, "solver_iter"))
        solver_iter = np.array([f(row, "solver_iter") for row in rows])
        mean_res = np.array([f(row, "mean_solver_residual") for row in rows])
        q10_res = np.array([f(row, "q10_solver_residual") for row in rows])
        q90_res = np.array([f(row, "q90_solver_residual") for row in rows])
        ax_d.semilogy(
            solver_iter,
            mean_res,
            marker="o",
            color=payload["color"],
            linewidth=1.7,
            label=payload["label"],
        )
        ax_d.fill_between(solver_iter, q10_res, q90_res, color=payload["color"], alpha=0.14)
    ax_d.set_xlabel("Anderson solver iteration")
    ax_d.set_ylabel("Mean solver residual")
    ax_d.set_ylim(0.007, 0.8)
    ax_d.legend(loc="upper right", fontsize=7, handlelength=1.4)
    ax_d.grid(axis="both", color="#e6e8eb", linewidth=0.7, which="major")
    ax_d.spines["left"].set_color("#666666")
    ax_d.spines["bottom"].set_color("#666666")
    panel_label(ax_d, "D")
    ax_d.set_title("Full-test convergence trace", loc="left", fontsize=8)

    fig.suptitle(
        "Sparse-PnP-ADMM-DEQ matches DEQ-MPI quality with faster paired inference",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.935,
        "Canonical paired full-test evaluation; deltas are Sparse-PnP minus DEQ-MPI on the same target/noise draws.",
        ha="left",
        va="top",
        fontsize=7,
        color="#4d5560",
    )

    base_path = out_dir / prefix
    save_figure(fig, base_path)
    plt.close(fig)

    notes_path = out_dir / f"{prefix}_notes.md"
    quality_delta_mean = paired_summary["canonical_delta_mean_db"]
    fast_delta_mean = fast_summary["fast_delta_mean_db"]
    fast_speedup = fast_summary["fast_speedup_mean"]
    eta_gain = eta_summary["gain_vs_eta0p10_db"]
    trace_final = {
        item["setting"]["name"]: item["trace"]["final_mean_solver_residual"]
        for item in trace_summary["settings"]
    }
    notes = [
        "# Sparse-PnP-DEQ Quality/Runtime Figure Notes",
        "",
        "## Figure Contract",
        "- Core conclusion: Sparse-PnP-ADMM-DEQ reaches DEQ-MPI-level reconstruction quality under paired canonical evaluation while offering a faster inference setting.",
        "- Archetype: quantitative grid.",
        "- Evidence chain: panel A shows quality-speed tradeoff; panel B tunes eta for fast inference; panel C validates quality and fast settings across seeds; panel D shows full-test Anderson solver residual traces.",
        "- Review risk: panel D residuals are batch-level Anderson residuals averaged across full-test batches, not per-sample residual trajectories.",
        "",
        "## Headline Numbers",
        f"- Quality setting canonical mean delta: `{quality_delta_mean:.4f} dB`.",
        f"- Fast setting canonical mean delta: `{fast_delta_mean:.4f} dB`.",
        f"- Fast setting mean speedup: `{fast_speedup:.2f}x`.",
        f"- Eta tuning gain at it10 over eta=0.10: `{eta_gain:.4f} dB`.",
        f"- Quality final mean solver residual: `{trace_final['quality_it12_eta0p10']:.6f}`.",
        f"- Fast final mean solver residual: `{trace_final['fast_it10_eta0p001']:.6f}`.",
        "",
        "## Outputs",
        f"- SVG: `{base_path.with_suffix('.svg')}`",
        f"- PDF: `{base_path.with_suffix('.pdf')}`",
        f"- PNG: `{base_path.with_suffix('.png')}`",
        f"- TIFF: `{base_path.with_suffix('.tiff')}`",
        f"- Source data: `{source_csv}`",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    outputs = {
        "svg": str(base_path.with_suffix(".svg")),
        "pdf": str(base_path.with_suffix(".pdf")),
        "png": str(base_path.with_suffix(".png")),
        "tiff": str(base_path.with_suffix(".tiff")),
        "source_data_csv": str(source_csv),
        "notes_md": str(notes_path),
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    outputs["summary_json"] = str(summary_path)
    return outputs


def main():
    args = parse_args()
    outputs = make_figure(Path(args.evalDir), Path(args.outDir), args.prefix)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
