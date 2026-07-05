"""Summarize OpenMPI 3D real-data validation artifacts.

Example:
  python make_openmpi_validation_summary.py

This script consumes existing OpenMPI reconstruction/metrics files and writes a
compact validation table, report, and metrics figure. It does not rerun
reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("training/reproduction")
PAPER_DIR = ROOT / "paper_figures"
DEFAULT_SELECTED_CSV = PAPER_DIR / "openmpi_selected_paper_results.csv"
DEFAULT_OUT_DIR = ROOT / "openmpi_validation"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize OpenMPI 3D validation results")
    parser.add_argument("--selectedCsv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--outDir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--prefix", default="openmpi_validation_summary")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_volume(path: Path) -> np.ndarray:
    data = np.load(path)
    vol = np.asarray(data["recon"], dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got {vol.shape}")
    if not np.isfinite(vol).all():
        raise ValueError(f"Non-finite values in {path}")
    return vol


def p(path_text: str) -> Path:
    return Path(path_text)


def fnum(row: dict, key: str) -> float:
    return float(row[key])


def volume_stats(vol: np.ndarray):
    peak = float(vol.max())
    q = np.quantile(vol, [0.5, 0.95, 0.99, 0.995])
    if peak <= 0:
        support_1pct = 0.0
        support_5pct = 0.0
        support_10pct = 0.0
    else:
        support_1pct = float((vol > 0.01 * peak).mean())
        support_5pct = float((vol > 0.05 * peak).mean())
        support_10pct = float((vol > 0.10 * peak).mean())
    return {
        "mean": float(vol.mean()),
        "max": peak,
        "median": float(q[0]),
        "p95": float(q[1]),
        "p99": float(q[2]),
        "p995": float(q[3]),
        "support_gt_1pct_peak": support_1pct,
        "support_gt_5pct_peak": support_5pct,
        "support_gt_10pct_peak": support_10pct,
        "nonzero_gt_1e-4": float((vol > 1e-4).mean()),
    }


def residual_summary(metrics: dict):
    history = metrics.get("history", [])
    if not history:
        return {
            "initial_residual": None,
            "final_residual": None,
            "final_over_initial_residual": None,
            "min_residual": None,
            "num_history": 0,
        }
    residuals = [float(item["data_residual"]) for item in history]
    first = residuals[0]
    final = residuals[-1]
    return {
        "initial_residual": first,
        "final_residual": final,
        "final_over_initial_residual": final / first if first > 0 else None,
        "min_residual": min(residuals),
        "num_history": len(residuals),
    }


def selected_rows(selected_csv: Path):
    rows = []
    for row in read_csv(selected_csv):
        vol_path = p(row["recon"])
        metrics_path = p(row["metrics"])
        vol = load_volume(vol_path)
        metrics = read_json(metrics_path)
        stats = volume_stats(vol)
        residuals = residual_summary(metrics)
        if row["method"] == "ZeroShot-PnP":
            method_family = f"{row['denoiser'].capitalize()} PnP"
        else:
            method_family = f"{row['denoiser'].capitalize()} l1-PnP"
        rows.append(
            {
                "phantom": row["phantom"],
                "scale": row["scale"],
                "method_label": row["method_label"],
                "method_family": method_family,
                "denoiser": row["denoiser"],
                "method": row["method"],
                "alpha": float(row["alpha"]),
                "rows": int(row["rows"]),
                "image_shape": "x".join(str(v) for v in metrics["image_shape"]),
                "nIter": int(metrics["nIter"]),
                "elapsed_seconds": float(metrics["elapsed_seconds"]),
                **stats,
                **residuals,
                "recon": str(vol_path),
                "metrics": str(metrics_path),
            }
        )
    return rows


def alpha_rows(root: Path = ROOT):
    paths = [
        root / "openmpi_shape_large_drunet_alpha_sweep.csv",
        root / "openmpi_medium_drunet_alpha_sweep_resolution_concentration.csv",
    ]
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            metrics = read_json(p(row["metrics"]))
            residuals = residual_summary(metrics)
            rows.append(
                {
                    "phantom": row["phantom"],
                    "scale": row.get("scale", "medium"),
                    "denoiser": row["denoiser"],
                    "alpha": float(row["alpha"]),
                    "mean": float(row["mean"]),
                    "max": float(row["max"]),
                    "p95": float(row["p95"]),
                    "p99": float(row["p99"]),
                    "nonzero_gt_1e-4": float(row["nonzero_gt_1e-4"]),
                    "final_residual": float(row["final_residual"]),
                    "final_over_initial_residual": residuals["final_over_initial_residual"],
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "metrics": row["metrics"],
                    "recon": row["recon"],
                }
            )
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def l1_effects(rows):
    effects = []
    for phantom in sorted({row["phantom"] for row in rows}):
        drunet_pnp = next(
            row
            for row in rows
            if row["phantom"] == phantom
            and row["denoiser"] == "drunet"
            and row["method"] == "ZeroShot-PnP"
        )
        drunet_l1 = next(
            row
            for row in rows
            if row["phantom"] == phantom
            and row["denoiser"] == "drunet"
            and row["method"] == "ZeroShot-l1-PnP"
        )
        effects.append(
            {
                "phantom": phantom,
                "p95_ratio_l1_over_pnp": drunet_l1["p95"] / drunet_pnp["p95"]
                if drunet_pnp["p95"] > 0
                else None,
                "p99_ratio_l1_over_pnp": drunet_l1["p99"] / drunet_pnp["p99"]
                if drunet_pnp["p99"] > 0
                else None,
                "support_5pct_delta": drunet_l1["support_gt_5pct_peak"]
                - drunet_pnp["support_gt_5pct_peak"],
                "residual_ratio_delta": drunet_l1["final_over_initial_residual"]
                - drunet_pnp["final_over_initial_residual"],
            }
        )
    return effects


def key_findings(rows, alpha):
    elapsed = [row["elapsed_seconds"] for row in rows]
    residual_ratio = [
        row["final_over_initial_residual"]
        for row in rows
        if row["final_over_initial_residual"] is not None
    ]
    effects = l1_effects(rows)
    p95_ratios = [item["p95_ratio_l1_over_pnp"] for item in effects]
    support_deltas = [item["support_5pct_delta"] for item in effects]
    alpha_best_residual = {}
    for phantom in sorted({row["phantom"] for row in alpha}):
        subset = [row for row in alpha if row["phantom"] == phantom]
        best = min(subset, key=lambda row: row["final_residual"])
        alpha_best_residual[phantom] = {
            "alpha": best["alpha"],
            "final_residual": best["final_residual"],
        }
    return {
        "num_selected_recons": len(rows),
        "phantoms": sorted({row["phantom"] for row in rows}),
        "elapsed_median_seconds": float(np.median(elapsed)),
        "elapsed_max_seconds": float(np.max(elapsed)),
        "residual_final_over_initial_min": float(np.min(residual_ratio)),
        "residual_final_over_initial_max": float(np.max(residual_ratio)),
        "drunet_l1_p95_ratio_min": float(np.min(p95_ratios)),
        "drunet_l1_p95_ratio_max": float(np.max(p95_ratios)),
        "drunet_l1_support_5pct_delta_min": float(np.min(support_deltas)),
        "drunet_l1_support_5pct_delta_max": float(np.max(support_deltas)),
        "alpha_best_residual": alpha_best_residual,
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
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def save_figure(rows, alpha, path_base: Path):
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
    colors = {
        "Gaussian PnP": "#4e79a7",
        "Gaussian l1-PnP": "#59a89c",
        "Drunet PnP": "#f28e2b",
        "Drunet l1-PnP": "#d65f5f",
    }
    phantoms = ["shape", "resolution", "concentration"]
    methods = ["Gaussian PnP", "Gaussian l1-PnP", "Drunet PnP", "Drunet l1-PnP"]
    x = np.arange(len(phantoms))

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.11, top=0.88, wspace=0.32, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))
    for idx, method in enumerate(methods):
        vals = [
            next(row for row in rows if row["phantom"] == phantom and row["method_family"] == method)[
                "final_residual"
            ]
            for phantom in phantoms
        ]
        ax_a.bar(x + offsets[idx], vals, width=width, label=method, color=colors.get(method), alpha=0.82)
    ax_a.set_yscale("log")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([p.capitalize() for p in phantoms])
    ax_a.set_ylabel("Final data residual")
    ax_a.set_title("Data consistency scale", loc="left", fontsize=8)
    style_axes(ax_a)
    panel_label(ax_a, "A")

    for idx, method in enumerate(methods):
        vals = [
            next(row for row in rows if row["phantom"] == phantom and row["method_family"] == method)[
                "support_gt_5pct_peak"
            ]
            for phantom in phantoms
        ]
        ax_b.bar(x + offsets[idx], vals, width=width, label=method, color=colors.get(method), alpha=0.82)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([p.capitalize() for p in phantoms])
    ax_b.set_ylabel("Support >5% peak")
    ax_b.set_title("Relative support", loc="left", fontsize=8)
    style_axes(ax_b)
    panel_label(ax_b, "B")
    ax_b.legend(frameon=False, fontsize=6, loc="upper right")

    alpha_colors = {"shape": "#4e79a7", "resolution": "#59a89c", "concentration": "#f28e2b"}
    for phantom in phantoms:
        subset = sorted([row for row in alpha if row["phantom"] == phantom], key=lambda row: row["alpha"])
        p99_ref = max(subset[0]["p99"], 1e-12)
        ax_c.plot(
            [row["alpha"] for row in subset],
            [row["p99"] / p99_ref for row in subset],
            marker="o",
            linewidth=1.6,
            color=alpha_colors[phantom],
            label=f"{phantom} p99",
        )
        ax_c.plot(
            [row["alpha"] for row in subset],
            [row["final_residual"] / max(subset[0]["final_residual"], 1e-12) for row in subset],
            marker="s",
            linewidth=1.2,
            linestyle="--",
            color=alpha_colors[phantom],
            alpha=0.58,
            label=f"{phantom} residual rel.",
        )
    ax_c.set_xscale("log")
    ax_c.set_xlabel("DRUNet l1 alpha")
    ax_c.set_ylabel("Relative p99 / residual")
    ax_c.set_title("Alpha sweep trends", loc="left", fontsize=8)
    style_axes(ax_c)
    panel_label(ax_c, "C")
    ax_c.legend(frameon=False, fontsize=6, ncol=3)

    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, rows, alpha, findings, effects):
    lines = [
        "# OpenMPI 3D Real-Data Validation",
        "",
        "- Scope: existing OpenMPI 3D real-data reconstructions for shape, resolution, and concentration phantoms.",
        "- Interpretation: no ground-truth PSNR is available; evidence is feasibility-oriented via visual structure, data residuals, intensity distribution, and support.",
        "",
        "## Key Findings",
        (
            f"- Selected reconstructions run quickly: median `{findings['elapsed_median_seconds']:.3f}s`, "
            f"max `{findings['elapsed_max_seconds']:.3f}s`."
        ),
        (
            f"- Final/first-recorded residual ratios span "
            f"`{findings['residual_final_over_initial_min']:.3f}` to "
            f"`{findings['residual_final_over_initial_max']:.3f}`; l1 regularization trades data fit for cleaner support."
        ),
        (
            f"- DRUNet-l1 changes p95 intensity by `{findings['drunet_l1_p95_ratio_min']:.3f}` to "
            f"`{findings['drunet_l1_p95_ratio_max']:.3f}x` relative to DRUNet PnP, depending on phantom."
        ),
        (
            f"- Relative support shifts by `{findings['drunet_l1_support_5pct_delta_min']:+.4f}` to "
            f"`{findings['drunet_l1_support_5pct_delta_max']:+.4f}` at the >5% peak threshold."
        ),
        "",
        "## Selected Reconstructions",
        "",
        "| Phantom | Method | Max | p95 | p99 | Support >5% peak | Final residual | Final/first-recorded | Recon |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['phantom']} | "
            f"{row['method_label']} | "
            f"{fmt(row['max'])} | "
            f"{fmt(row['p95'])} | "
            f"{fmt(row['p99'])} | "
            f"{fmt(row['support_gt_5pct_peak'])} | "
            f"{fmt(row['final_residual'], 2)} | "
            f"{fmt(row['final_over_initial_residual'], 3)} | "
            f"`{row['recon']}` |"
        )
    lines.extend(
        [
            "",
            "## DRUNet-l1 Effects",
            "",
            "| Phantom | p95 l1/PnP | p99 l1/PnP | Support delta | Residual-ratio delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            "| "
            f"{row['phantom']} | "
            f"{fmt(row['p95_ratio_l1_over_pnp'], 3)} | "
            f"{fmt(row['p99_ratio_l1_over_pnp'], 3)} | "
            f"{fmt(row['support_5pct_delta'], 4)} | "
            f"{fmt(row['residual_ratio_delta'], 3)} |"
        )
    lines.extend(["", "## Alpha Sweep Best Residual"])
    for phantom, payload in findings["alpha_best_residual"].items():
        lines.append(
            f"- `{phantom}`: alpha `{payload['alpha']:.0f}`, final residual `{payload['final_residual']:.3f}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(Path(args.selectedCsv))
    alpha = alpha_rows()
    effects = l1_effects(rows)
    findings = key_findings(rows, alpha)

    selected_csv = out_dir / f"{args.prefix}_selected.csv"
    alpha_csv = out_dir / f"{args.prefix}_alpha_sweep.csv"
    json_path = out_dir / f"{args.prefix}.json"
    md_path = out_dir / f"{args.prefix}.md"
    figure_base = out_dir / args.prefix

    write_csv(selected_csv, rows)
    if alpha:
        write_csv(alpha_csv, alpha)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_rows": rows,
                "alpha_rows": alpha,
                "drunet_l1_effects": effects,
                "key_findings": findings,
            },
            f,
            indent=2,
        )
    write_markdown(md_path, rows, alpha, findings, effects)
    save_figure(rows, alpha, figure_base)

    print(
        json.dumps(
            {
                "key_findings": findings,
                "outputs": {
                    "selected_csv": str(selected_csv),
                    "alpha_csv": str(alpha_csv),
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
