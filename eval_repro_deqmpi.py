import argparse
import json
import time

import torch
import torch.nn.functional as F
from scipy.io import loadmat

from data import MRAdatasetH5NoScale, loadMtxExp
from modelClasses import DEQFixedPoint, fixedPointTekrarlar, rdnLDFixedPt
from reconAlgos import ADMMfncs, psnr
from trainerClasses import admmInputGenerator, getNoisyData, transformDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a reproduced DEQ-MPI checkpoint on the simulated test set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pSNRval", type=float, default=10.0)
    parser.add_argument("--batchSize", type=int, default=256)
    parser.add_argument("--nbOfSingulars", type=int, default=220)
    parser.add_argument("--maxIter", type=int, default=25)
    parser.add_argument("--outJson", default="training/reproduction/deqmpi_eval_final.json")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    n1, n2 = 26, 13
    img_size = [n1, n2]
    n_img = n1 * n2

    sys_mtx_ref = loadMtxExp("inhouseData/expMatinHouse").reshape(-1, n_img)

    interpolater = loadmat("interpExp2.mat")["interpolater"]
    sys_mtx_hr_int2 = sys_mtx_ref @ torch.from_numpy(interpolater).float().to(device)
    sys_mtx = F.avg_pool2d(sys_mtx_hr_int2.reshape(sys_mtx_ref.shape[0], 2 * n1, 2 * n2), 2).reshape(
        sys_mtx_ref.shape[0], -1
    )

    u, s, vh = torch.linalg.svd(sys_mtx, full_matrices=False)
    u = u[:, : args.nbOfSingulars]
    s = s[: args.nbOfSingulars]
    v = vh[: args.nbOfSingulars, :].T

    test_data_hr = MRAdatasetH5NoScale("datasets/testPatches.h5", prefetch=True).data
    test_data = transformDataset(test_data_hr, [n1, n2], [0.5, 1], [0, 0])
    ref_vals = test_data.squeeze().reshape(test_data.shape[0], -1)

    my_data_nsless = getNoisyData(test_data, 0, sys_mtx_ref)
    my_data_lr_inv_cr = getNoisyData(test_data, 0, sys_mtx)
    std_val = 10 ** (-args.pSNRval / 20) * 0.41
    my_data_gen = getNoisyData(test_data, std_val, sys_mtx_ref)

    measured_snr_no_inverse_crime = float(
        20 * torch.log10(torch.norm(my_data_nsless) / torch.norm(my_data_gen - my_data_nsless))
    )
    measured_snr_inverse_crime = float(
        20 * torch.log10(torch.norm(my_data_lr_inv_cr) / torch.norm(my_data_gen - my_data_lr_inv_cr))
    )

    epsilon = float(std_val * (my_data_nsless.shape[2]) ** 0.5)
    datat_c_full = my_data_gen.reshape(test_data.shape[0], -1)
    comp_data, lsqr_inp = admmInputGenerator(my_data_gen, u, s, v, (-1, 1, n1, n2))
    x_in = lsqr_inp.reshape(-1, n1, n2)

    at_c = sys_mtx.reshape(-1, n_img)
    admm = ADMMfncs(at_c, args.maxIter, 100, img_size, diagnose=False)
    madmm = admm.MtC

    comp_for_ls = F.linear(datat_c_full.reshape(test_data.shape[0], -1), u.T)
    val_inp = F.linear(comp_for_ls / (s + 1e-4), v).reshape(-1, 1, n1, n2)

    model_core = rdnLDFixedPt(
        1,
        12,
        4,
        4,
        12,
        8,
        1,
        1,
        bias=True,
        numDim=2,
        consistencyDim=1,
    ).to(device)
    model = DEQFixedPoint(model_core, fixedPointTekrarlar, max_iter=args.maxIter).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    n_batch = datat_c_full.shape[0]
    d0 = torch.zeros_like(val_inp).reshape(n_batch, -1)
    d2 = torch.zeros_like(datat_c_full)
    model_in = torch.cat((val_inp.reshape(n_batch, -1), d0, d2), dim=1)
    x_rec = torch.zeros_like(ref_vals).reshape(-1, 1, n1, n2)

    start_time = time.time()
    with torch.no_grad():
        for start in range(0, n_batch, args.batchSize):
            end = min(start + args.batchSize, n_batch)
            batch_shape = (end - start, 1, n1, n2)
            fixed_pts = (datat_c_full[start:end], at_c, madmm, epsilon, batch_shape)
            x_rec[start:end] = model(model_in[start:end], fixed_pts)[:, :n_img].reshape(batch_shape)

    x_rec_flat = x_rec.reshape(n_batch, -1)
    deq_psnr = psnr(ref_vals, x_rec_flat)
    lsqr_psnr = psnr(ref_vals, x_in.reshape(n_batch, -1))

    result = {
        "checkpoint": args.checkpoint,
        "num_test_images": int(n_batch),
        "pSNRval": args.pSNRval,
        "measured_snr_no_inverse_crime_db": measured_snr_no_inverse_crime,
        "measured_snr_inverse_crime_db": measured_snr_inverse_crime,
        "epsilon": epsilon,
        "nbOfSingulars": args.nbOfSingulars,
        "maxIter": args.maxIter,
        "mean_lsqr_psnr_db": float(lsqr_psnr.mean()),
        "mean_deq_psnr_db": float(deq_psnr.mean()),
        "std_deq_psnr_db": float(deq_psnr.std()),
        "min_deq_psnr_db": float(deq_psnr.min()),
        "max_deq_psnr_db": float(deq_psnr.max()),
        "elapsed_seconds": time.time() - start_time,
    }

    print(json.dumps(result, indent=2))
    with open(args.outJson, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
