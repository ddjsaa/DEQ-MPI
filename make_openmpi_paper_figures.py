import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("training/reproduction")
OUT_DIR = ROOT / "paper_figures"


MAIN_CONFIGS = {
    "shape": {
        "scale": "large",
        "methods": [
            ("Gaussian PnP", "openmpi_shape_large_zeroshot_pnp_gaussian"),
            ("Gaussian l1, a=5000", "openmpi_shape_large_zeroshot_l1_pnp_gaussian_alpha5000"),
            ("DRUNet PnP", "openmpi_shape_large_zeroshot_pnp_drunet"),
            ("DRUNet l1, a=10000", "openmpi_shape_large_zeroshot_l1_pnp_drunet_alpha10000"),
        ],
    },
    "resolution": {
        "scale": "medium",
        "methods": [
            ("Gaussian PnP", "openmpi_resolution_medium_zeroshot_pnp_gaussian"),
            ("Gaussian l1, a=5000", "openmpi_resolution_medium_zeroshot_l1_pnp_gaussian_alpha5000"),
            ("DRUNet PnP", "openmpi_resolution_medium_zeroshot_pnp_drunet"),
            ("DRUNet l1, a=10000", "openmpi_resolution_medium_zeroshot_l1_pnp_drunet_alpha10000"),
        ],
    },
    "concentration": {
        "scale": "medium",
        "methods": [
            ("Gaussian PnP", "openmpi_concentration_medium_zeroshot_pnp_gaussian"),
            ("Gaussian l1, a=5000", "openmpi_concentration_medium_zeroshot_l1_pnp_gaussian_alpha5000"),
            ("DRUNet PnP", "openmpi_concentration_medium_zeroshot_pnp_drunet"),
            ("DRUNet l1, a=5000", "openmpi_concentration_medium_zeroshot_l1_pnp_drunet_alpha5000"),
        ],
    },
}


ALPHA_CONFIGS = {
    "shape": ("large", [1000, 2500, 5000, 10000]),
    "resolution": ("medium", [1000, 2500, 5000, 10000]),
    "concentration": ("medium", [1000, 2500, 5000, 10000]),
}


def load_volume(stem: str) -> np.ndarray:
    path = ROOT / f"{stem}_recon.npz"
    data = np.load(path)
    vol = data["recon"].astype(np.float32)
    if vol.ndim != 3 or not np.isfinite(vol).all():
        raise ValueError(f"Invalid reconstruction: {path}, shape={vol.shape}")
    return vol


def load_metrics(stem: str) -> dict:
    path = ROOT / f"{stem}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def axial(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def mip(volume: np.ndarray) -> np.ndarray:
    return volume.max(axis=0)


def save_main_panel():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phantoms = list(MAIN_CONFIGS.keys())
    n_rows = len(phantoms) * 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9), constrained_layout=True)

    for p_idx, phantom in enumerate(phantoms):
        methods = MAIN_CONFIGS[phantom]["methods"]
        volumes = [(label, load_volume(stem)) for label, stem in methods]
        vmax = max(float(v.max()) for _, v in volumes) or 1.0
        for col, (label, vol) in enumerate(volumes):
            for view_idx, (view_name, image) in enumerate((("axial", axial(vol)), ("MIP", mip(vol)))):
                row = p_idx * 2 + view_idx
                ax = axes[row, col]
                im = ax.imshow(image, cmap="magma", vmin=0, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(label, fontsize=9)
                if col == 0:
                    scale = MAIN_CONFIGS[phantom]["scale"]
                    ax.set_ylabel(f"{phantom} {scale}\n{view_name}", fontsize=9)
        fig.colorbar(im, ax=axes[p_idx * 2 : p_idx * 2 + 2, :].ravel().tolist(), shrink=0.65)

    out = OUT_DIR / "openmpi_main_selected_methods.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_alpha_panel():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ALPHA_CONFIGS), 4, figsize=(11, 7), constrained_layout=True)

    for row, (phantom, (scale, alphas)) in enumerate(ALPHA_CONFIGS.items()):
        vols = []
        for alpha in alphas:
            stem = f"openmpi_{phantom}_{scale}_zeroshot_l1_pnp_drunet_alpha{alpha}"
            vols.append((alpha, load_volume(stem)))
        vmax = max(float(v.max()) for _, v in vols) or 1.0
        for col, (alpha, vol) in enumerate(vols):
            ax = axes[row, col]
            im = ax.imshow(mip(vol), cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"alpha={alpha}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{phantom} {scale}\nMIP", fontsize=9)
        fig.colorbar(im, ax=axes[row, :].ravel().tolist(), shrink=0.72)

    out = OUT_DIR / "openmpi_drunet_alpha_selection_mip.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def selected_results_csv():
    rows = []
    for phantom, cfg in MAIN_CONFIGS.items():
        for label, stem in cfg["methods"]:
            vol = load_volume(stem)
            metrics = load_metrics(stem)
            q = np.quantile(vol, [0.5, 0.95, 0.99])
            rows.append(
                {
                    "phantom": phantom,
                    "scale": cfg["scale"],
                    "method_label": label,
                    "denoiser": metrics["denoiser"],
                    "method": metrics["method"],
                    "alpha": metrics["alpha"],
                    "rows": metrics["A_shape"][0],
                    "mean": float(vol.mean()),
                    "max": float(vol.max()),
                    "median": float(q[0]),
                    "p95": float(q[1]),
                    "p99": float(q[2]),
                    "nonzero_gt_1e-4": float((vol > 1e-4).mean()),
                    "final_residual": metrics["history"][-1]["data_residual"],
                    "recon": str(ROOT / f"{stem}_recon.npz"),
                    "metrics": str(ROOT / f"{stem}_metrics.json"),
                }
            )

    out = OUT_DIR / "openmpi_selected_paper_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main():
    print(save_main_panel())
    print(save_alpha_panel())
    print(selected_results_csv())


if __name__ == "__main__":
    main()
