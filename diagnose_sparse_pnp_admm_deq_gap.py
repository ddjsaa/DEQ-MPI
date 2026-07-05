"""Diagnose per-sample PSNR gaps for Sparse-PnP-ADMM-DEQ.

The script joins a Sparse-PnP checkpoint evaluation CSV with the paper-figure
per-sample baseline CSV, then writes combined metrics, tail tables, quantile-bin
summaries, and a short Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_RUN_DIR = Path(
    "training/reproduction/sparse_pnp_admm_deq/"
    "deq_anderson_rdn_full_e3_it12_eta0p10"
)
DEFAULT_SPARSE_CSV = DEFAULT_RUN_DIR / "evaluations/test_full_best_sample_metrics.csv"
DEFAULT_SPARSE_SUMMARY = DEFAULT_RUN_DIR / "evaluations/test_full_best_summary.json"
DEFAULT_PAPER_CSV = Path(
    "training/reproduction/paper_figures/deqmpi_vs_zeroshot_2d_all_sample_metrics.csv"
)
DEFAULT_PAPER_SUMMARY = Path(
    "training/reproduction/paper_figures/deqmpi_vs_zeroshot_2d_summary.json"
)
DEFAULT_CANONICAL_DEQ = Path("training/reproduction/deqmpi_eval_epoch200END.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Sparse-PnP vs DEQ-MPI sample gaps")
    parser.add_argument("--sparseCsv", default=str(DEFAULT_SPARSE_CSV))
    parser.add_argument("--sparseSummary", default=str(DEFAULT_SPARSE_SUMMARY))
    parser.add_argument("--paperCsv", default=str(DEFAULT_PAPER_CSV))
    parser.add_argument("--paperSummary", default=str(DEFAULT_PAPER_SUMMARY))
    parser.add_argument("--canonicalDeqSummary", default=str(DEFAULT_CANONICAL_DEQ))
    parser.add_argument("--outDir", default=str(DEFAULT_RUN_DIR / "diagnostics"))
    parser.add_argument("--prefix", default="test_full_sparse_vs_deqmpi_seed2026")
    parser.add_argument("--topN", type=int, default=50)
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_sparse_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["sample_index"])
            rows[idx] = {
                "sample_index": idx,
                "sparse_psnr_db": float(row["psnr_db"]),
                "sparse_nrmse": float(row["nrmse"]),
                "sparse_data_residual_norm": float(row["data_residual_norm"]),
                "sparse_nonzero_ratio": float(row["nonzero_ratio"]),
                "sparse_min_value": float(row["min_value"]),
                "sparse_max_value": float(row["max_value"]),
            }
    return rows


def read_paper_rows(path: Path) -> dict[int, dict[str, float]]:
    by_sample: dict[int, dict[str, float]] = {}
    method_names = {
        "deq_mpi": "deq_mpi_psnr_db",
        "ls_svd": "ls_svd_psnr_db",
        "zeroshot_pnp": "zeroshot_pnp_psnr_db",
        "zeroshot_l1_pnp": "zeroshot_l1_pnp_psnr_db",
    }
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method not in method_names:
                continue
            idx = int(row["sample_index"])
            by_sample.setdefault(idx, {"sample_index": idx})[method_names[method]] = float(
                row["psnr_db"]
            )
    return by_sample


def join_rows(sparse_rows: dict[int, dict[str, float]], paper_rows: dict[int, dict[str, float]]):
    joined = []
    for idx in sorted(set(sparse_rows) & set(paper_rows)):
        row = dict(sparse_rows[idx])
        row.update(paper_rows[idx])
        required = ["deq_mpi_psnr_db", "ls_svd_psnr_db"]
        if not all(key in row for key in required):
            continue
        row["delta_sparse_minus_deq_mpi_db"] = row["sparse_psnr_db"] - row["deq_mpi_psnr_db"]
        row["gap_deq_mpi_minus_sparse_db"] = -row["delta_sparse_minus_deq_mpi_db"]
        row["delta_sparse_minus_ls_svd_db"] = row["sparse_psnr_db"] - row["ls_svd_psnr_db"]
        if "zeroshot_pnp_psnr_db" in row:
            row["delta_sparse_minus_zeroshot_pnp_db"] = (
                row["sparse_psnr_db"] - row["zeroshot_pnp_psnr_db"]
            )
        if "zeroshot_l1_pnp_psnr_db" in row:
            row["delta_sparse_minus_zeroshot_l1_pnp_db"] = (
                row["sparse_psnr_db"] - row["zeroshot_l1_pnp_psnr_db"]
            )
        joined.append(row)
    return joined


def values(rows: Iterable[dict[str, float]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def stats(vals: np.ndarray) -> dict[str, float]:
    if vals.size == 0:
        return {}
    quantiles = np.quantile(vals, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "q25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "q75": float(quantiles[5]),
        "q90": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
        "max": float(vals.max()),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation_summary(rows: list[dict[str, float]], target_field: str):
    target = values(rows, target_field)
    fields = [
        "sparse_psnr_db",
        "deq_mpi_psnr_db",
        "ls_svd_psnr_db",
        "sparse_nrmse",
        "sparse_data_residual_norm",
        "sparse_nonzero_ratio",
        "sparse_min_value",
        "sparse_max_value",
    ]
    out = {}
    for field in fields:
        if field not in rows[0]:
            continue
        corr = pearson(values(rows, field), target)
        if corr is not None:
            out[field] = corr
    return dict(sorted(out.items(), key=lambda item: abs(item[1]), reverse=True))


def make_bins(rows: list[dict[str, float]], bin_fields: list[str], n_bins: int = 5):
    out = []
    delta = values(rows, "delta_sparse_minus_deq_mpi_db")
    for field in bin_fields:
        x = values(rows, field)
        edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
        assignments = np.searchsorted(edges[1:-1], x, side="right")
        for bin_idx in range(n_bins):
            mask = assignments == bin_idx
            if not np.any(mask):
                continue
            sub_rows = [rows[i] for i in np.flatnonzero(mask)]
            sub_delta = delta[mask]
            out.append(
                {
                    "bin_field": field,
                    "bin": bin_idx + 1,
                    "low": float(np.min(x[mask])),
                    "high": float(np.max(x[mask])),
                    "count": int(np.sum(mask)),
                    "mean_sparse_psnr_db": float(values(sub_rows, "sparse_psnr_db").mean()),
                    "mean_deq_mpi_psnr_db": float(values(sub_rows, "deq_mpi_psnr_db").mean()),
                    "mean_ls_svd_psnr_db": float(values(sub_rows, "ls_svd_psnr_db").mean()),
                    "mean_delta_sparse_minus_deq_mpi_db": float(sub_delta.mean()),
                    "median_delta_sparse_minus_deq_mpi_db": float(np.median(sub_delta)),
                    "win_rate_vs_deq_mpi": float(np.mean(sub_delta > 0.0)),
                    "mean_sparse_nrmse": float(values(sub_rows, "sparse_nrmse").mean()),
                    "mean_sparse_data_residual_norm": float(
                        values(sub_rows, "sparse_data_residual_norm").mean()
                    ),
                }
            )
    return out


def tail_summary(rows: list[dict[str, float]], fractions=(0.05, 0.10, 0.20)):
    ordered = sorted(rows, key=lambda row: row["delta_sparse_minus_deq_mpi_db"])
    out = {}
    for frac in fractions:
        n = max(1, math.ceil(len(ordered) * frac))
        worst = ordered[:n]
        best = ordered[-n:]
        out[f"worst_{int(frac * 100)}pct"] = {
            "count": n,
            "mean_delta_sparse_minus_deq_mpi_db": float(
                values(worst, "delta_sparse_minus_deq_mpi_db").mean()
            ),
            "mean_sparse_psnr_db": float(values(worst, "sparse_psnr_db").mean()),
            "mean_deq_mpi_psnr_db": float(values(worst, "deq_mpi_psnr_db").mean()),
            "mean_ls_svd_psnr_db": float(values(worst, "ls_svd_psnr_db").mean()),
            "mean_sparse_data_residual_norm": float(
                values(worst, "sparse_data_residual_norm").mean()
            ),
        }
        out[f"best_{int(frac * 100)}pct"] = {
            "count": n,
            "mean_delta_sparse_minus_deq_mpi_db": float(
                values(best, "delta_sparse_minus_deq_mpi_db").mean()
            ),
            "mean_sparse_psnr_db": float(values(best, "sparse_psnr_db").mean()),
            "mean_deq_mpi_psnr_db": float(values(best, "deq_mpi_psnr_db").mean()),
            "mean_ls_svd_psnr_db": float(values(best, "ls_svd_psnr_db").mean()),
            "mean_sparse_data_residual_norm": float(
                values(best, "sparse_data_residual_norm").mean()
            ),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, object]]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(path: Path, summary: dict, bin_rows: list[dict[str, object]], outputs: dict):
    delta = summary["metrics"]["delta_sparse_minus_deq_mpi_db"]
    sparse = summary["metrics"]["sparse_psnr_db"]
    deq = summary["metrics"]["deq_mpi_psnr_db"]
    tail = summary["tail"]
    correlations = summary["correlations"]["delta_sparse_minus_deq_mpi_db"]
    worst_bins = sorted(
        bin_rows,
        key=lambda row: float(row["mean_delta_sparse_minus_deq_mpi_db"]),
    )[:6]

    lines = [
        "# Sparse-PnP-ADMM-DEQ Gap Diagnostics",
        "",
        "## Sources",
        f"- Sparse CSV: `{summary['inputs']['sparse_csv']}`",
        f"- DEQ-MPI baseline CSV: `{summary['inputs']['paper_csv']}`",
        f"- Joined samples: `{summary['num_joined_samples']}`",
        "",
        "## Headline",
        (
            f"- Sparse-PnP mean PSNR: `{format_float(sparse['mean'])} dB`; "
            f"DEQ-MPI(seed=2026) mean PSNR: `{format_float(deq['mean'])} dB`."
        ),
        (
            f"- Mean delta Sparse-PnP minus DEQ-MPI: `{format_float(delta['mean'])} dB`; "
            f"median `{format_float(delta['median'])} dB`, q05/q95 "
            f"`{format_float(delta['q05'])} / {format_float(delta['q95'])} dB`."
        ),
        (
            f"- Sparse-PnP wins on `{format_float(summary['win_rates']['delta_gt_0'] * 100, 2)}%` "
            f"of samples and is within +/-0.1 dB on "
            f"`{format_float(summary['win_rates']['abs_delta_lte_0p1'] * 100, 2)}%`."
        ),
        "",
        "## Tail Shape",
        (
            f"- Worst 5% mean delta: "
            f"`{format_float(tail['worst_5pct']['mean_delta_sparse_minus_deq_mpi_db'])} dB` "
            f"over `{tail['worst_5pct']['count']}` samples."
        ),
        (
            f"- Best 5% mean delta: "
            f"`{format_float(tail['best_5pct']['mean_delta_sparse_minus_deq_mpi_db'])} dB` "
            f"over `{tail['best_5pct']['count']}` samples."
        ),
        "",
        "## Strongest Correlations With Delta",
    ]
    for field, corr in list(correlations.items())[:6]:
        lines.append(f"- `{field}`: `{format_float(corr)}`")

    lines.extend(["", "## Weakest Quantile Bins"])
    for row in worst_bins:
        lines.append(
            "- "
            f"`{row['bin_field']}` bin {row['bin']} "
            f"[{format_float(float(row['low']))}, {format_float(float(row['high']))}]: "
            f"mean delta `{format_float(float(row['mean_delta_sparse_minus_deq_mpi_db']))} dB`, "
            f"win rate `{format_float(float(row['win_rate_vs_deq_mpi']) * 100, 2)}%`"
        )

    if summary.get("canonical_deq_mpi_mean_psnr_db") is not None:
        lines.extend(
            [
                "",
                "## Canonical Reference",
                (
                    "- The historical fixed-report DEQ-MPI baseline mean is "
                    f"`{format_float(summary['canonical_deq_mpi_mean_psnr_db'])} dB`; "
                    "the seed=2026 per-sample baseline used for this join is "
                    f"`{format_float(deq['mean'])} dB`."
                ),
            ]
        )

    lines.extend(["", "## Outputs"])
    for name, out_path in outputs.items():
        lines.append(f"- `{name}`: `{out_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    sparse_csv = Path(args.sparseCsv)
    paper_csv = Path(args.paperCsv)
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse_rows = read_sparse_rows(sparse_csv)
    paper_rows = read_paper_rows(paper_csv)
    rows = join_rows(sparse_rows, paper_rows)
    if not rows:
        raise RuntimeError("No joined sample rows. Check sample_index coverage.")

    top_n = min(args.topN, len(rows))
    worst_rows = sorted(rows, key=lambda row: row["delta_sparse_minus_deq_mpi_db"])[:top_n]
    best_rows = sorted(rows, key=lambda row: row["delta_sparse_minus_deq_mpi_db"], reverse=True)[
        :top_n
    ]
    bin_fields = [
        "ls_svd_psnr_db",
        "deq_mpi_psnr_db",
        "sparse_psnr_db",
        "sparse_nrmse",
        "sparse_data_residual_norm",
        "sparse_max_value",
    ]
    bin_rows = make_bins(rows, bin_fields)

    delta_vals = values(rows, "delta_sparse_minus_deq_mpi_db")
    sparse_summary = read_json(Path(args.sparseSummary))
    paper_summary = read_json(Path(args.paperSummary))
    canonical_deq = read_json(Path(args.canonicalDeqSummary))

    metrics = {}
    metric_fields = [
        "sparse_psnr_db",
        "deq_mpi_psnr_db",
        "ls_svd_psnr_db",
        "zeroshot_pnp_psnr_db",
        "zeroshot_l1_pnp_psnr_db",
        "delta_sparse_minus_deq_mpi_db",
        "gap_deq_mpi_minus_sparse_db",
        "delta_sparse_minus_ls_svd_db",
        "delta_sparse_minus_zeroshot_pnp_db",
        "delta_sparse_minus_zeroshot_l1_pnp_db",
        "sparse_nrmse",
        "sparse_data_residual_norm",
    ]
    for field in metric_fields:
        if field in rows[0]:
            metrics[field] = stats(values(rows, field))

    outputs = {
        "combined_csv": str(out_dir / f"{args.prefix}_combined.csv"),
        "summary_json": str(out_dir / f"{args.prefix}_summary.json"),
        "bin_summary_csv": str(out_dir / f"{args.prefix}_bin_summary.csv"),
        "worst_gap_csv": str(out_dir / f"{args.prefix}_worst_gap_top{top_n}.csv"),
        "best_gap_csv": str(out_dir / f"{args.prefix}_best_gap_top{top_n}.csv"),
        "report_md": str(out_dir / f"{args.prefix}_report.md"),
    }

    summary = {
        "inputs": {
            "sparse_csv": str(sparse_csv),
            "sparse_summary": args.sparseSummary,
            "paper_csv": str(paper_csv),
            "paper_summary": args.paperSummary,
            "canonical_deq_summary": args.canonicalDeqSummary,
        },
        "num_sparse_samples": len(sparse_rows),
        "num_paper_samples": len(paper_rows),
        "num_joined_samples": len(rows),
        "metrics": metrics,
        "win_rates": {
            "delta_gt_0": float(np.mean(delta_vals > 0.0)),
            "delta_gt_0p1": float(np.mean(delta_vals > 0.1)),
            "delta_gt_1p0": float(np.mean(delta_vals > 1.0)),
            "delta_lt_minus_0p1": float(np.mean(delta_vals < -0.1)),
            "delta_lt_minus_1p0": float(np.mean(delta_vals < -1.0)),
            "abs_delta_lte_0p1": float(np.mean(np.abs(delta_vals) <= 0.1)),
            "abs_delta_lte_0p5": float(np.mean(np.abs(delta_vals) <= 0.5)),
        },
        "tail": tail_summary(rows),
        "correlations": {
            "delta_sparse_minus_deq_mpi_db": correlation_summary(
                rows, "delta_sparse_minus_deq_mpi_db"
            )
        },
        "sparse_summary": sparse_summary,
        "paper_summary": paper_summary,
        "canonical_deq_mpi_mean_psnr_db": (
            canonical_deq.get("mean_deq_psnr_db") if canonical_deq else None
        ),
        "outputs": outputs,
    }

    write_csv(Path(outputs["combined_csv"]), rows)
    write_csv(Path(outputs["bin_summary_csv"]), bin_rows)
    write_csv(Path(outputs["worst_gap_csv"]), worst_rows)
    write_csv(Path(outputs["best_gap_csv"]), best_rows)
    with Path(outputs["summary_json"]).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(Path(outputs["report_md"]), summary, bin_rows, outputs)

    print(
        json.dumps(
            {
                "num_joined_samples": len(rows),
                "mean_sparse_psnr_db": metrics["sparse_psnr_db"]["mean"],
                "mean_deq_mpi_psnr_db": metrics["deq_mpi_psnr_db"]["mean"],
                "mean_delta_sparse_minus_deq_mpi_db": metrics[
                    "delta_sparse_minus_deq_mpi_db"
                ]["mean"],
                "win_rate_vs_deq_mpi": summary["win_rates"]["delta_gt_0"],
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
