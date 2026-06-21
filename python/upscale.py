#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "realesrgan==0.3.0",
#   "opencv-python-headless==4.11.0.86",
#   "torchvision==0.15.2",
#   "numpy==1.26.4",
# ]
# ///

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

MODELS = {
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "blocks": 23,
    },
    "x4v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "file": "realesr-general-x4v3.pth",
        "blocks": 6,
    },
    "x4v3-denoise": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "file": "realesr-general-wdn-x4v3.pth",
        "blocks": 6,
    },
}


def _model_path(key: str) -> str:
    info = MODELS[key]
    cache = Path.home() / ".cache" / "realesrgan"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / info["file"]
    if not path.exists():
        print(f"downloading {info['file']}...")
        torch.hub.download_url_to_file(info["url"], str(path))
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upscale an image 4x using Real-ESRGAN")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, nargs="?", help="Output image path (default: input_upscaled.ext)")
    parser.add_argument("--model", choices=list(MODELS), default="x4plus", help="Model variant (default: x4plus)")
    parser.add_argument("--denoise", type=float, default=0, help="Denoise strength 0-1 (x4v3-denoise only)")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_stem(args.input.stem + "_upscaled")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    info = MODELS[args.model]
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=info["blocks"], num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=_model_path(args.model), model=model, device=device)

    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read {args.input}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    result, _ = upsampler.enhance(img, outscale=4)
    elapsed = time.perf_counter() - t0

    cv2.imwrite(str(output), result)
    print(f"wrote {output} ({result.shape[1]}x{result.shape[0]}) — {elapsed:.1f}s")


if __name__ == "__main__":
    main()