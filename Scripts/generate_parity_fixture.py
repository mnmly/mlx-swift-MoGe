"""Generate a parity fixture for end-to-end MoGe-2 inference.

Loads the Torch reference (`Ruicheng/moge-2-vitl-normal`), runs
`MoGeModel.infer` on a real image, and dumps the input + final outputs to a
`.safetensors` file that the Swift `ParityFixtureTests` consumes.

We only check end-to-end — there's no raw-forward fixture and no separate
encoder check. If the final dict (`points`, `depth`, `normal`, `mask`,
`intrinsics`) matches within tolerances on the same weights and image, the
port is considered correct.

Run from the python/MoGe venv (it needs `moge`, `torch`, `cv2`,
`safetensors`):

    /path/to/MoGe/.venv/bin/python Scripts/generate_parity_fixture.py \
        --image /path/to/MoGe/example_images/01_HouseIndoor.jpg \
        --out Tests/Fixtures
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

from moge.model.v2 import MoGeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Torch MoGe-2 parity fixture.")
    parser.add_argument("--hf-model", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out", required=True, help="Output fixture directory.")
    parser.add_argument("--device", default="mps", help="Torch device.")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=280,
        help="Resize the input to this width (height follows aspect). 0 keeps the original.",
    )
    parser.add_argument("--resolution-level", type=int, default=5)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = not args.no_fp16

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if args.resize_width and args.resize_width > 0:
        h, w = rgb.shape[:2]
        new_w = args.resize_width
        new_h = int(round(h * (new_w / w)))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    H, W = rgb.shape[:2]
    arr_nhwc = (rgb.astype(np.float32) / 255.0)            # (H, W, 3) fp32 [0, 1]
    image_nhwc = torch.from_numpy(arr_nhwc).contiguous()   # CPU
    image_chw = image_nhwc.permute(2, 0, 1).contiguous().to(args.device)

    model = MoGeModel.from_pretrained(args.hf_model).to(args.device).eval()
    if use_fp16:
        model = model.half()

    with torch.inference_mode():
        out = model.infer(
            image_chw,
            resolution_level=args.resolution_level,
            force_projection=True,
            apply_mask=True,
            use_fp16=use_fp16,
        )
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device.startswith("cuda"):
            torch.cuda.synchronize()

    tensors: dict[str, torch.Tensor] = {
        "input_nhwc": image_nhwc.float().cpu(),
    }
    for key in ("points", "depth", "normal", "mask", "intrinsics"):
        if key in out and out[key] is not None:
            tensors[f"output.{key}"] = out[key].detach().float().cpu().contiguous()

    # `apply_mask=True` sets invalid regions of points/depth to +inf. Replace
    # those with NaN before serialization — safetensors round-trips NaN cleanly,
    # and the Swift test masks them out before computing tolerances.
    for k in ("output.points", "output.depth"):
        if k in tensors:
            t = tensors[k]
            t[~torch.isfinite(t)] = float("nan")
            tensors[k] = t

    fixture_path = out_dir / "moge_infer.safetensors"
    save_file(tensors, str(fixture_path))

    metadata = {
        "hf_model": args.hf_model,
        "device": args.device,
        "image": str(Path(args.image).resolve()),
        "height": H,
        "width": W,
        "resolution_level": args.resolution_level,
        "fp16": use_fp16,
        "keys": sorted(tensors.keys()),
    }
    (out_dir / "moge_infer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {fixture_path}")
    print(f"Input: {W}x{H}, resolution_level={args.resolution_level}, fp16={use_fp16}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
