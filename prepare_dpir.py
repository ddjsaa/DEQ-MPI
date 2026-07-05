import argparse
import subprocess
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare the official DPIR DRUNet dependency for ZeroShot-PnP.")
    parser.add_argument("--dpirRoot", default="external/DPIR")
    parser.add_argument("--modelDir", default="external/models")
    parser.add_argument("--modelName", default="drunet_gray.pth")
    parser.add_argument("--repo", default="https://github.com/cszn/DPIR.git")
    parser.add_argument(
        "--modelUrl",
        default="https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth",
    )
    args = parser.parse_args()

    dpir_root = Path(args.dpirRoot)
    model_dir = Path(args.modelDir)
    model_path = model_dir / args.modelName

    if not dpir_root.exists():
        dpir_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", args.repo, str(dpir_root)], check=True)
    else:
        print(f"DPIR already exists: {dpir_root}")

    if not model_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.modelUrl}")
        urllib.request.urlretrieve(args.modelUrl, model_path)
        print(f"Saved {model_path}")
    else:
        print(f"DRUNet weights already exist: {model_path}")

    print("Ready. Use:")
    print(
        "  python eval_zeroshot_l1_pnp.py --denoiser drunet "
        f"--dpirRoot {dpir_root} --drunetWeights {model_path}"
    )


if __name__ == "__main__":
    main()
