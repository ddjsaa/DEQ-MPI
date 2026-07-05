import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIG_DIR = Path("training/reproduction/paper_figures")
TWO_D_SUMMARY = FIG_DIR / "deqmpi_vs_zeroshot_2d_summary.json"
OPENMPI_SELECTED = FIG_DIR / "openmpi_selected_paper_results.csv"


def load_2d_summary():
    with TWO_D_SUMMARY.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_openmpi_rows():
    with OPENMPI_SELECTED.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw_box(ax, xy, wh, title, body, fc, ec="#2c3440"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=10.2, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.02, body, ha="center", va="center", fontsize=7.6, linespacing=1.18)
    return patch


def arrow(ax, start, end, color="#344054", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def make_method_diagram(out_path):
    fig, ax = plt.subplots(figsize=(14.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "input": "#eef6ff",
        "deq": "#eaf7ea",
        "zs": "#fff3df",
        "hybrid": "#f2ecff",
        "out": "#edf2f7",
    }

    draw_box(ax, (0.035, 0.43), (0.16, 0.17), "MPI data", "measurement y\nsystem matrix A\nnoise level epsilon", colors["input"])
    draw_box(ax, (0.245, 0.63), (0.22, 0.18), "DEQ-MPI backbone", "SVD/LS initialization\nADMM state q=[x,d0,d2]\nfixed-point solver", colors["deq"])
    draw_box(ax, (0.535, 0.66), (0.18, 0.15), "RDN prior", "learned image prior\nR_theta(x-d0)", colors["deq"])
    draw_box(ax, (0.535, 0.43), (0.18, 0.15), "Learned consistency", "C_phi(Ax-d2, y, epsilon)", colors["deq"])

    draw_box(ax, (0.245, 0.18), (0.22, 0.19), "Zero-shot prior path", "DRUNet D_sigma\nl1 / TV proximal prior\nnonnegative projection", colors["zs"])
    draw_box(ax, (0.535, 0.18), (0.18, 0.16), "ACM", "epsilon-ball projection\nresidual-driven sigma\nadaptive beta/gamma", colors["zs"])

    draw_box(ax, (0.785, 0.47), (0.18, 0.19), "ZS-DEQ-MPI", "DEQ-MPI fixed point\n+ ZSRB image prior\n+ ACM data consistency", colors["hybrid"])
    draw_box(ax, (0.785, 0.18), (0.18, 0.15), "Output", "MPI concentration map\n2D benchmark\n3D OpenMPI transfer", colors["out"])

    arrow(ax, (0.195, 0.53), (0.245, 0.72))
    arrow(ax, (0.195, 0.48), (0.245, 0.28))
    arrow(ax, (0.465, 0.72), (0.535, 0.735))
    arrow(ax, (0.465, 0.70), (0.535, 0.505))
    arrow(ax, (0.465, 0.275), (0.535, 0.26))
    arrow(ax, (0.715, 0.735), (0.785, 0.60))
    arrow(ax, (0.715, 0.505), (0.785, 0.565))
    arrow(ax, (0.715, 0.26), (0.785, 0.50))
    arrow(ax, (0.875, 0.47), (0.875, 0.33))

    ax.text(
        0.50,
        0.93,
        "ZS-DEQ-MPI: zero-shot regularized deep-equilibrium reconstruction for MPI",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.885,
        "The proposed route keeps the DEQ-MPI fixed-point backbone and inserts zero-shot priors and adaptive physics consistency.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475467",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_2d_csv(summary, out_path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "mean_psnr_db",
                "std_psnr_db",
                "min_psnr_db",
                "max_psnr_db",
                "elapsed_seconds",
                "training_category",
            ],
        )
        writer.writeheader()
        category = {
            "LS/SVD": "physics only",
            "DEQ-MPI": "supervised MPI training",
            "ZeroShot-PnP": "generic pretrained denoiser, no MPI training",
            "ZeroShot-l1-PnP": "generic pretrained denoiser plus explicit l1 prior",
        }
        for row in summary["summary"]:
            writer.writerow({**row, "training_category": category.get(row["method"], "")})


def write_openmpi_csv(rows, out_path):
    selected = [
        row
        for row in rows
        if row["denoiser"] == "drunet"
        and row["method_label"] in {"DRUNet PnP", "DRUNet l1, a=5000", "DRUNet l1, a=10000"}
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phantom",
                "scale",
                "method_label",
                "alpha",
                "max",
                "p99",
                "nonzero_gt_1e-4",
                "final_residual",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "phantom": row["phantom"],
                    "scale": row["scale"],
                    "method_label": row["method_label"],
                    "alpha": row["alpha"],
                    "max": row["max"],
                    "p99": row["p99"],
                    "nonzero_gt_1e-4": row["nonzero_gt_1e-4"],
                    "final_residual": row["final_residual"],
                }
            )


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_tables_md(summary, openmpi_rows, out_path):
    two_d_rows = []
    for row in summary["summary"]:
        two_d_rows.append(
            [
                row["method"],
                f"{row['mean_psnr_db']:.3f}",
                f"{row['std_psnr_db']:.3f}",
                f"{row['elapsed_seconds']:.3f}",
            ]
        )

    drunet_rows = [
        row
        for row in openmpi_rows
        if row["denoiser"] == "drunet"
        and row["method_label"] in {"DRUNet PnP", "DRUNet l1, a=5000", "DRUNet l1, a=10000"}
    ]
    open_rows = []
    for row in drunet_rows:
        open_rows.append(
            [
                f"{row['phantom']} ({row['scale']})",
                row["method_label"],
                row["alpha"],
                f"{float(row['p99']):.4f}",
                f"{float(row['nonzero_gt_1e-4']):.3f}",
                f"{float(row['final_residual']):.1f}",
            ]
        )

    text = f"""# ZS-DEQ-MPI paper-ready result tables

## Table 1. Matched 2D simulated benchmark

All metrics are generated from `deqmpi_vs_zeroshot_2d_summary.json` using all {summary['num_test_images']} test images.

{markdown_table(['Method', 'Mean PSNR (dB)', 'Std (dB)', 'Runtime (s)'], two_d_rows)}

## Table 2. Public 3D OpenMPI zero-shot transfer summary

These rows summarize the DRUNet-based zero-shot reconstructions. OpenMPI rows are qualitative/transfer evidence, not a direct comparison against the 2D DEQ-MPI checkpoint.

{markdown_table(['Phantom', 'Method', 'alpha', 'p99', 'Nonzero ratio', 'Final residual'], open_rows)}

## Key Interpretation

The matched 2D benchmark confirms that supervised DEQ-MPI remains the strongest in-domain method. The zero-shot variants recover the dominant tracer morphology with no MPI-specific training of the denoiser, but their lower PSNR shows the expected accuracy-transfer tradeoff. The l1 proximal step makes reconstructions more sparse and can reduce weak structures, which is useful to discuss as a data-dependent structural prior rather than as a universal PSNR booster.
"""
    out_path.write_text(text, encoding="utf-8")


def write_bridge_text(summary, out_path):
    rows = {row["method"]: row for row in summary["summary"]}
    deq = rows["DEQ-MPI"]
    pnp = rows["ZeroShot-PnP"]
    l1 = rows["ZeroShot-l1-PnP"]

    text = f"""# Manuscript bridge text for ZS-DEQ-MPI

## Suggested Results Paragraph

On the matched simulated 2D benchmark, the reproduced DEQ-MPI checkpoint achieved {deq['mean_psnr_db']:.2f} +/- {deq['std_psnr_db']:.2f} dB over {summary['num_test_images']} test images, confirming that the supervised deep-equilibrium baseline remains the strongest in-domain reconstruction method. The DRUNet-based ZeroShot-PnP branch reached {pnp['mean_psnr_db']:.2f} +/- {pnp['std_psnr_db']:.2f} dB under the same measurement model, while the ZeroShot-l1-PnP variant reached {l1['mean_psnr_db']:.2f} +/- {l1['std_psnr_db']:.2f} dB. These results indicate that the zero-shot branch does not replace supervised DEQ-MPI on the matched 2D setting. Instead, it provides a training-light regularization path that preserves the fixed-point reconstruction philosophy while reducing the dependence on MPI-specific paired training data.

## Suggested Discussion Paragraph

The l1 proximal step slightly reduced PSNR on the 2D vessel benchmark, suggesting that the sparsity prior can suppress low-intensity structures when the tracer distribution is not strongly sparse. This behavior is consistent with the role of l1/TV terms as structural priors whose benefit depends on the target distribution and noise regime. For this reason, we position ZS-DEQ-MPI as a hybrid framework rather than a single fixed prior: DEQ-MPI supplies a strong learned in-domain fixed point, while the zero-shot denoiser, l1/TV proximal operators, and adaptive consistency module provide controllable mechanisms for low-training and out-of-distribution deployment.

## Recommended Figure Callout

Figure X shows representative 2D reconstructions selected after excluding low-energy ground-truth patches and sampling DEQ-MPI PSNR quantiles. DEQ-MPI produces the most faithful in-domain reconstructions, whereas ZeroShot-PnP retains the dominant tracer morphology with smoother appearance. Adding the l1 proximal step further increases sparsity, which may be advantageous for sparse tracer targets but can reduce PSNR in this dense simulated-vessel setting.

## Limitation Statement

The current OpenMPI experiments should be interpreted as zero-shot 3D transfer evidence. They are not a direct 3D comparison against DEQ-MPI, because the reproduced DEQ-MPI checkpoint was trained for the 2D simulated vessel setting and is not directly compatible with the public 3D OpenMPI system matrix without architectural adaptation and retraining or recalibration.
"""
    out_path.write_text(text, encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_2d_summary()
    openmpi_rows = load_openmpi_rows()

    make_method_diagram(FIG_DIR / "zs_deqmpi_method_diagram.png")
    write_2d_csv(summary, FIG_DIR / "zs_deqmpi_2d_summary_table.csv")
    write_openmpi_csv(openmpi_rows, FIG_DIR / "zs_deqmpi_openmpi_summary_table.csv")
    write_tables_md(summary, openmpi_rows, FIG_DIR / "zs_deqmpi_result_tables.md")
    write_bridge_text(summary, FIG_DIR / "zs_deqmpi_manuscript_bridge.md")

    print(
        json.dumps(
            {
                "method_diagram": str(FIG_DIR / "zs_deqmpi_method_diagram.png"),
                "two_d_table": str(FIG_DIR / "zs_deqmpi_2d_summary_table.csv"),
                "openmpi_table": str(FIG_DIR / "zs_deqmpi_openmpi_summary_table.csv"),
                "tables_markdown": str(FIG_DIR / "zs_deqmpi_result_tables.md"),
                "bridge_text": str(FIG_DIR / "zs_deqmpi_manuscript_bridge.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
