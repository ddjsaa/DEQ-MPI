import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_recon(path: str | Path) -> np.ndarray:
    data = np.load(path)
    if "recon" not in data:
        raise KeyError(f"{path} does not contain a 'recon' array.")
    vol = np.asarray(data["recon"], dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D reconstruction, got shape {vol.shape} from {path}.")
    if not np.isfinite(vol).all():
        raise ValueError(f"{path} contains non-finite values.")
    return vol


def center_slices_and_mips(volume: np.ndarray):
    z, y, x = (s // 2 for s in volume.shape)
    return [
        ("axial", volume[z, :, :]),
        ("coronal", volume[:, y, :]),
        ("sagittal", volume[:, :, x]),
        ("mip", volume.max(axis=0)),
    ]


def save_single_views(recon_path: str | Path, out_dir: str | Path, cmap: str = "magma"):
    recon_path = Path(recon_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = load_recon(recon_path)
    vmax = float(volume.max()) or 1.0
    outputs = []

    for name, image in center_slices_and_mips(volume):
        fig, ax = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
        im = ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.78)
        out = out_dir / f"{recon_path.stem}_{name}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    return outputs


def save_phantom_grid(phantom: str, out_path: str | Path, root: str | Path, cmap: str = "magma"):
    root = Path(root)
    configs = [
        ("Gaussian PnP", f"openmpi_{phantom}_medium_zeroshot_pnp_gaussian_recon.npz"),
        ("Gaussian l1-PnP", f"openmpi_{phantom}_medium_zeroshot_l1_pnp_gaussian_alpha5000_recon.npz"),
        ("DRUNet PnP", f"openmpi_{phantom}_medium_zeroshot_pnp_drunet_recon.npz"),
        ("DRUNet l1-PnP", f"openmpi_{phantom}_medium_zeroshot_l1_pnp_drunet_alpha5000_recon.npz"),
    ]
    volumes = [(label, load_recon(root / filename)) for label, filename in configs]
    vmax = max(float(vol.max()) for _, vol in volumes) or 1.0
    views = ["axial", "coronal", "sagittal", "mip"]

    fig, axes = plt.subplots(len(views), len(volumes), figsize=(11, 9), constrained_layout=True)
    for col, (label, volume) in enumerate(volumes):
        view_map = dict(center_slices_and_mips(volume))
        for row, view in enumerate(views):
            ax = axes[row, col]
            im = ax.imshow(view_map[view], cmap=cmap, vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(view, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export 3D OpenMPI reconstruction slices and MIPs.")
    parser.add_argument("--recon", default=None, help="Single *_recon.npz file to export.")
    parser.add_argument("--phantom", choices=["shape", "resolution", "concentration"], default=None)
    parser.add_argument("--root", default="training/reproduction")
    parser.add_argument("--outDir", default="training/reproduction/openmpi_views")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cmap", default="magma")
    args = parser.parse_args()

    if args.recon:
        outputs = save_single_views(args.recon, args.outDir, cmap=args.cmap)
        for path in outputs:
            print(path)
    if args.phantom:
        out = args.out or str(Path(args.outDir) / f"openmpi_{args.phantom}_medium_views_grid.png")
        print(save_phantom_grid(args.phantom, out, args.root, cmap=args.cmap))
    if not args.recon and not args.phantom:
        raise SystemExit("Pass --recon or --phantom.")


if __name__ == "__main__":
    main()
