import argparse
import json
import time

import torch
import torch.nn.functional as F
from scipy.io import loadmat

from data import MRAdatasetH5NoScale, loadMtxExp
from reconAlgos import psnr
from trainerClasses import getNoisyData, transformDataset
from zeroshot_l1_pnp import ZeroShotL1PnP, build_denoiser


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ZeroShot-l1-PnP on the DEQ-MPI reproduction simulated test set."
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--nIter", type=int, default=7)
    parser.add_argument("--mu0", type=float, default=2e5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--testLimit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--directSolve", action="store_true", default=True)
    parser.add_argument("--noDirectSolve", dest="directSolve", action="store_false")
    parser.add_argument("--outJson", default="training/reproduction/zeroshot_l1_pnp_eval.json")

    parser.add_argument("--denoiser", choices=["gaussian", "drunet"], default="gaussian")
    parser.add_argument("--gaussianKernel", type=int, default=5)
    parser.add_argument("--gaussianSigma", type=float, default=1.0)
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--drunetWeights", default="external/models/drunet_gray.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    n1, n2 = 26, 13
    img_size = [n1, n2]
    n_img = n1 * n2

    sys_mtx_ref = loadMtxExp("inhouseData/expMatinHouse").reshape(-1, n_img)

    interpolater = loadmat("interpExp2.mat")["interpolater"]
    sys_mtx_hr_int2 = sys_mtx_ref @ torch.from_numpy(interpolater).float().to(device)
    sys_mtx = F.avg_pool2d(sys_mtx_hr_int2.reshape(sys_mtx_ref.shape[0], 2 * n1, 2 * n2), 2).reshape(
        sys_mtx_ref.shape[0], -1
    )

    # Match the DEQ-MPI evaluation path: use SVD only to form the reduced
    # measurement model, not as a learned component of the proposed method.
    u, s, vh = torch.linalg.svd(sys_mtx, full_matrices=False)
    u = u[:, : args.nbOfSingulars]
    s = s[: args.nbOfSingulars]
    v = vh[: args.nbOfSingulars, :].T
    A_reduced = torch.diag(s) @ v.T

    test_data_hr = MRAdatasetH5NoScale("datasets/testPatches.h5", prefetch=True).data
    test_data = transformDataset(test_data_hr, img_size, [0.5, 1], [0, 0])
    if args.testLimit and args.testLimit > 0:
        test_data = test_data[: args.testLimit]
    ref_vals = test_data.squeeze().reshape(test_data.shape[0], -1)

    my_data_nsless = getNoisyData(test_data, 0, sys_mtx_ref)
    std_val = 10 ** (-args.pSNRval / 20) * 0.41
    my_data_gen = getNoisyData(test_data, std_val, sys_mtx_ref)
    measured_snr = float(20 * torch.log10(torch.norm(my_data_nsless) / torch.norm(my_data_gen - my_data_nsless)))

    y_reduced = my_data_gen.reshape(test_data.shape[0], -1) @ u

    denoiser = build_denoiser(args, device)
    reconstructor = ZeroShotL1PnP(
        A_reduced.to(device),
        img_size,
        denoiser,
        n_iter=args.nIter,
        mu0=args.mu0,
        alpha=args.alpha,
        direct_solve=args.directSolve,
        non_negative=True,
    )

    x_rec = torch.zeros_like(ref_vals)
    histories = []
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, y_reduced.shape[0], args.batchSize):
            end = min(start + args.batchSize, y_reduced.shape[0])
            out, hist = reconstructor(y_reduced[start:end], return_history=True)
            x_rec[start:end] = out
            if not histories:
                histories = hist

    zs_psnr = psnr(ref_vals, x_rec)
    result = {
        "method": "ZeroShot-l1-PnP",
        "denoiser": args.denoiser,
        "num_test_images": int(y_reduced.shape[0]),
        "seed": args.seed,
        "pSNRval": args.pSNRval,
        "measured_snr_db": measured_snr,
        "nbOfSingulars": args.nbOfSingulars,
        "nIter": args.nIter,
        "mu0": args.mu0,
        "alpha": 0.005 * args.mu0 if args.alpha is None else args.alpha,
        "mean_psnr_db": float(zs_psnr.mean()),
        "std_psnr_db": float(zs_psnr.std()),
        "min_psnr_db": float(zs_psnr.min()),
        "max_psnr_db": float(zs_psnr.max()),
        "elapsed_seconds": time.time() - start_time,
        "first_batch_history": histories,
    }

    print(json.dumps(result, indent=2))
    with open(args.outJson, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
