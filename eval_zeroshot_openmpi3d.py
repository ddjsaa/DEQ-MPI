import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from zeroshot_l1_pnp import ZeroShotL1PnP, build_denoiser


def main():
    parser = argparse.ArgumentParser(description="Run ZeroShot-PnP on a converted 3D OpenMPI npz system.")
    parser.add_argument("--npz", required=True, help="NPZ with A, y, image_shape from openmpi3d_tools.py convert.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nIter", type=int, default=7)
    parser.add_argument("--mu0", type=float, default=5e5)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--cg", action="store_true", help="Use CG instead of direct normal-equation solve.")
    parser.add_argument("--outRecon", default=None)
    parser.add_argument("--outJson", default=None)
    parser.add_argument(
        "--noNormalizeSystem",
        dest="normalizeSystem",
        action="store_false",
        help="Disable RMS scaling of raw OpenMPI A/y values.",
    )
    parser.set_defaults(normalizeSystem=True)

    parser.add_argument("--denoiser", choices=["gaussian", "drunet"], default="drunet")
    parser.add_argument("--gaussianKernel", type=int, default=5)
    parser.add_argument("--gaussianSigma", type=float, default=1.0)
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    data = np.load(args.npz)
    A_np = data["A"].astype(np.float32)
    y_np = data["y"].astype(np.float32)
    image_shape = tuple(int(v) for v in data["image_shape"])
    if np.prod(image_shape) != A_np.shape[1]:
        raise ValueError(f"image_shape {image_shape} does not match A columns {A_np.shape[1]}")
    if not np.isfinite(A_np).all() or not np.isfinite(y_np).all():
        raise ValueError("Input system contains non-finite values.")

    system_scale = 1.0
    if args.normalizeSystem:
        system_scale = float(np.sqrt(np.mean(A_np**2)))
        if not np.isfinite(system_scale) or system_scale <= 0:
            raise ValueError(f"Invalid system scale: {system_scale}")
        A_np = A_np / system_scale
        y_np = y_np / system_scale

    A = torch.from_numpy(A_np).to(device)
    y = torch.from_numpy(y_np).reshape(1, -1).to(device)

    denoiser = build_denoiser(args, device)
    reconstructor = ZeroShotL1PnP(
        A,
        image_shape,
        denoiser,
        n_iter=args.nIter,
        mu0=args.mu0,
        alpha=args.alpha,
        direct_solve=not args.cg,
        non_negative=True,
    )

    start = time.time()
    with torch.no_grad():
        recon, history = reconstructor(y, return_history=True)
    elapsed = time.time() - start
    recon_volume = recon.reshape(image_shape).detach().cpu().numpy()

    out_recon = args.outRecon or str(Path(args.npz).with_name(Path(args.npz).stem + "_recon.npz"))
    np.savez_compressed(out_recon, recon=recon_volume, history=np.asarray(history, dtype=object))

    result = {
        "method": "ZeroShot-l1-PnP" if args.alpha else "ZeroShot-PnP",
        "npz": args.npz,
        "denoiser": args.denoiser,
        "A_shape": list(A_np.shape),
        "image_shape": list(image_shape),
        "nIter": args.nIter,
        "mu0": args.mu0,
        "alpha": args.alpha,
        "normalizeSystem": args.normalizeSystem,
        "system_scale": system_scale,
        "elapsed_seconds": elapsed,
        "recon_min": float(recon_volume.min()),
        "recon_max": float(recon_volume.max()),
        "recon_mean": float(recon_volume.mean()),
        "history": history,
        "outRecon": out_recon,
    }
    print(json.dumps(result, indent=2))
    out_json = args.outJson or str(Path(args.npz).with_name(Path(args.npz).stem + "_recon_metrics.json"))
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
