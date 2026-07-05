import argparse
import json
import os
import ssl
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import torch


OPENMPI_BASE_URL = "https://media.tuhh.de/ibi/openMPIData/data"
OPENMPI_FILES = {
    "calibration": ("calibrations/3.mdf", "calibration_3.mdf"),
    "shape": ("measurements/shapePhantom/3.mdf", "shape_3.mdf"),
    "resolution": ("measurements/resolutionPhantom/3.mdf", "resolution_3.mdf"),
    "concentration": ("measurements/concentrationPhantom/3.mdf", "concentration_3.mdf"),
}


def download_file(url: str, out_path: Path, insecure: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        cmd = [curl, "-L", "--fail", "--continue-at", "-", "--output", str(out_path), url]
        if insecure:
            cmd.insert(1, "-k")
        subprocess.run(cmd, check=True)
        return

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    headers = {}
    mode = "wb"
    existing = 0
    if tmp.exists():
        existing = tmp.stat().st_size
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

    request = urllib.request.Request(url, headers=headers)
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(request, context=context) as response, open(tmp, mode) as f:
        total = response.headers.get("Content-Length")
        total = int(total) + existing if total is not None else None
        downloaded = existing
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r{out_path.name}: {downloaded / total * 100:5.1f}%", end="")
        print()
    tmp.replace(out_path)


def download_openmpi3d(root: str | Path, files: Iterable[str], insecure: bool = False):
    root = Path(root)
    for key in files:
        remote, local = OPENMPI_FILES[key]
        out_path = root / local
        url = f"{OPENMPI_BASE_URL}/{remote}"
        if out_path.exists():
            try:
                size = remote_size(url, insecure=insecure)
            except Exception:
                size = None
            if size is not None and out_path.stat().st_size == size:
                print(f"complete: {out_path}")
                continue
            print(f"resuming partial file: {out_path}")
        else:
            size = None
        if size is not None and out_path.exists() and out_path.stat().st_size > size:
            raise RuntimeError(f"Local file is larger than remote file: {out_path}")
        download_file(url, out_path, insecure=insecure)


def remote_size(url: str, insecure: bool = False) -> int | None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        cmd = [curl, "-L", "-sI", url]
        if insecure:
            cmd.insert(1, "-k")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                return int(line.split(":", 1)[1].strip())
        return None
    request = urllib.request.Request(url, method="HEAD")
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(request, context=context) as response:
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else None


def openmpi3d_status(root: str | Path, insecure: bool = False):
    root = Path(root)
    rows = []
    for key, (remote, local) in OPENMPI_FILES.items():
        url = f"{OPENMPI_BASE_URL}/{remote}"
        local_path = root / local
        local_size = local_path.stat().st_size if local_path.exists() else 0
        try:
            size = remote_size(url, insecure=insecure)
        except Exception:
            size = None
        rows.append(
            {
                "key": key,
                "local": str(local_path),
                "local_mb": round(local_size / 1024 / 1024, 2),
                "remote_mb": round(size / 1024 / 1024, 2) if size else None,
                "complete": bool(size and local_size == size),
            }
        )
    return rows


def complex_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields and "r" in arr.dtype.fields and "i" in arr.dtype.fields:
        return arr["r"] + 1j * arr["i"]
    return arr


def visit_hdf5(path: str | Path):
    rows = []
    with h5py.File(path, "r") as h5:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append(
                    {
                        "path": "/" + name,
                        "shape": tuple(int(v) for v in obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )
        h5.visititems(visitor)
    return rows


def read_scalar(h5: h5py.File, path: str, default=None):
    if path not in h5:
        return default
    value = h5[path][()]
    if np.ndim(value) == 0:
        return value.item()
    return value


def frequency_vector(meas_h5: h5py.File, n_time: int) -> np.ndarray:
    bandwidth = read_scalar(meas_h5, "/acquisition/receiver/bandwidth")
    if bandwidth is None:
        bandwidth = read_scalar(meas_h5, "/acquisition/receiver/bandwidthSampling")
    if bandwidth is None:
        raise KeyError("Could not find receiver bandwidth in MDF file.")
    num_freq = n_time // 2 + 1
    return np.arange(num_freq, dtype=np.float64) / (num_freq - 1) * float(bandwidth)


def load_minimal_openmpi_system(
    calibration_path: str | Path,
    measurement_path: str | Path,
    min_freq_hz: float = 80e3,
    max_rows: int | None = None,
    receive_channels: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Load a minimal background-free real-valued OpenMPI linear system.

    Returns A_real, y_real, image_shape. Complex rows are split into real and
    imaginary rows so the concentration vector remains real-valued.
    """
    with h5py.File(calibration_path, "r") as f_cal, h5py.File(measurement_path, "r") as f_meas:
        cal_ds = f_cal["/measurement/data"]
        meas_ds = f_meas["/measurement/data"]
        image_shape = tuple(int(v) for v in f_cal["/calibration/size"][()])

        if cal_ds.ndim != 4 or cal_ds.shape[0] != 1:
            raise ValueError(f"Expected calibration data shape [1, rx, freq, frames], got {cal_ds.shape}")
        if meas_ds.ndim != 4 or meas_ds.shape[1] != 1:
            raise ValueError(f"Expected measurement data shape [frames, 1, rx, time], got {meas_ds.shape}")

        n_rx = min(receive_channels, cal_ds.shape[1], meas_ds.shape[2])
        n_time = meas_ds.shape[-1]
        freq = frequency_vector(f_meas, n_time)
        n_freq = min(cal_ds.shape[2], n_time // 2 + 1, freq.shape[0])
        freq_indices = np.flatnonzero(freq[:n_freq] > min_freq_hz)
        if freq_indices.size == 0:
            raise ValueError(f"No frequency components above min_freq_hz={min_freq_hz}.")
        if max_rows is not None and max_rows > 0:
            n_freq_keep = min(freq_indices.size, int(np.ceil(max_rows / n_rx)))
            freq_indices = freq_indices[:n_freq_keep]

        cal_bg = None
        if "/measurement/isBackgroundFrame" in f_cal:
            cal_bg = f_cal["/measurement/isBackgroundFrame"][()].astype(bool).squeeze()

        # Read only the frequency rows needed for this reconstruction. This
        # keeps smoke tests from loading the full multi-GB calibration matrix.
        calib = complex_array(cal_ds[0, :n_rx, freq_indices, :])
        if cal_bg is not None and cal_bg.shape[0] == calib.shape[-1]:
            calib = calib[:, :, ~cal_bg]

        meas_frame_indices = np.arange(meas_ds.shape[0])
        if "/measurement/isBackgroundFrame" in f_meas:
            meas_bg = f_meas["/measurement/isBackgroundFrame"][()].astype(bool).squeeze()
            if meas_bg.shape[0] == meas_frame_indices.shape[0]:
                meas_frame_indices = meas_frame_indices[~meas_bg]
        if meas_frame_indices.size == 0:
            raise ValueError("No non-background measurement frames found.")

        meas_sum = np.zeros((n_rx, freq_indices.size), dtype=np.complex128)
        frame_count = 0
        chunk_frames = 64
        for start in range(0, meas_frame_indices.size, chunk_frames):
            chunk_idx = meas_frame_indices[start : start + chunk_frames]
            chunk = complex_array(meas_ds[chunk_idx, 0, :n_rx, :]).astype(np.float32, copy=False)
            chunk_freq = np.fft.rfft(chunk, axis=-1)[:, :, freq_indices]
            meas_sum += chunk_freq.sum(axis=0)
            frame_count += chunk_freq.shape[0]
        meas = meas_sum / frame_count

        A_complex = calib.reshape(n_rx * freq_indices.size, calib.shape[-1])
        y_complex = meas.reshape(n_rx * freq_indices.size)
        if max_rows is not None and max_rows > 0:
            A_complex = A_complex[:max_rows]
            y_complex = y_complex[:max_rows]

        A_real = np.concatenate([A_complex.real, A_complex.imag], axis=0).astype(np.float32)
        y_real = np.concatenate([y_complex.real, y_complex.imag], axis=0).astype(np.float32)
        return A_real, y_real, image_shape


def save_npz(out_path: str | Path, A: np.ndarray, y: np.ndarray, image_shape: Tuple[int, int, int]):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, A=A, y=y, image_shape=np.asarray(image_shape, dtype=np.int64))
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Download, inspect, and convert 3D OpenMPI MDF data.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--root", default="datasets/OpenMPI")
    p_dl.add_argument("--files", default="calibration,shape,resolution,concentration")
    p_dl.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for downloads.")

    p_ins = sub.add_parser("inspect")
    p_ins.add_argument("path")
    p_ins.add_argument("--outJson", default=None)

    p_conv = sub.add_parser("convert")
    p_conv.add_argument("--root", default="datasets/OpenMPI")
    p_conv.add_argument("--phantom", choices=["shape", "resolution", "concentration"], default="shape")
    p_conv.add_argument("--minFreqHz", type=float, default=80e3)
    p_conv.add_argument("--maxRows", type=int, default=0)
    p_conv.add_argument("--out", default=None)

    p_status = sub.add_parser("status")
    p_status.add_argument("--root", default="datasets/OpenMPI")
    p_status.add_argument("--insecure", action="store_true")

    args = parser.parse_args()
    if args.cmd == "download":
        files = [item.strip() for item in args.files.split(",") if item.strip()]
        download_openmpi3d(args.root, files, insecure=args.insecure)
    elif args.cmd == "inspect":
        rows = visit_hdf5(args.path)
        print(json.dumps(rows, indent=2))
        if args.outJson:
            Path(args.outJson).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif args.cmd == "convert":
        root = Path(args.root)
        calib = root / OPENMPI_FILES["calibration"][1]
        meas = root / OPENMPI_FILES[args.phantom][1]
        out = args.out or str(root / f"{args.phantom}_minimal_real_system.npz")
        A, y, shape = load_minimal_openmpi_system(
            calib,
            meas,
            min_freq_hz=args.minFreqHz,
            max_rows=args.maxRows if args.maxRows > 0 else None,
        )
        print("A", A.shape, A.dtype, "y", y.shape, y.dtype, "image_shape", shape)
        save_npz(out, A, y, shape)
    elif args.cmd == "status":
        print(json.dumps(openmpi3d_status(args.root, insecure=args.insecure), indent=2))


if __name__ == "__main__":
    main()
