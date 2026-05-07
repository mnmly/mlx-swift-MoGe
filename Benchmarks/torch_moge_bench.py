"""Benchmark Torch MoGe-2 inference for comparison against the mlx-swift port.

Mirrors `Tools/moge-bench/MoGeBench.swift`: same warmup/iterations protocol,
same resolution_level, same `apply_mask`/`force_projection`, and reports both
inference-only and end-to-end (load image -> tensor -> infer) timings.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from moge.model.v2 import MoGeModel


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_image_tensor(path: str, device: str) -> torch.Tensor:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Torch MoGe-2 inference.")
    parser.add_argument(
        "--model",
        default="Ruicheng/moge-2-vitl-normal",
        help="HF repo id or local model directory.",
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--device", default="mps", help="Torch device, e.g. cuda, mps, cpu.")
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 autocast (default: fp16).")
    parser.add_argument("--no-force-projection", action="store_true")
    parser.add_argument("--no-apply-mask", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--include-load",
        action="store_true",
        help="Include image load + tensor upload in measured time.",
    )
    args = parser.parse_args()

    image_path = str(Path(args.input).expanduser())
    use_fp16 = not args.no_fp16
    force_projection = not args.no_force_projection
    apply_mask = not args.no_apply_mask

    load_start = time.perf_counter()
    model = MoGeModel.from_pretrained(args.model).to(args.device).eval()
    if use_fp16:
        model = model.half()
    load_s = time.perf_counter() - load_start

    image_tensor = load_image_tensor(image_path, args.device)
    H, W = image_tensor.shape[-2:]

    @torch.inference_mode()
    def run_once_infer_only() -> None:
        out = model.infer(
            image_tensor,
            num_tokens=args.num_tokens,
            resolution_level=args.resolution_level,
            force_projection=force_projection,
            apply_mask=apply_mask,
            use_fp16=use_fp16,
        )
        # Touch a result to ensure materialization before sync.
        _ = out["points"]
        synchronize(args.device)

    @torch.inference_mode()
    def run_once_with_load() -> None:
        t = load_image_tensor(image_path, args.device)
        out = model.infer(
            t,
            num_tokens=args.num_tokens,
            resolution_level=args.resolution_level,
            force_projection=force_projection,
            apply_mask=apply_mask,
            use_fp16=use_fp16,
        )
        _ = out["points"]
        synchronize(args.device)

    run_once = run_once_with_load if args.include_load else run_once_infer_only

    for _ in range(max(0, args.warmup)):
        run_once()

    times: list[float] = []
    for _ in range(max(1, args.iterations)):
        synchronize(args.device)
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)

    print("backend=torch")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print(f"input={image_path}")
    print(f"source_size={W}x{H}")
    print(f"fp16={use_fp16}")
    print(f"resolution_level={args.resolution_level}")
    print(f"num_tokens={args.num_tokens}")
    print(f"force_projection={force_projection}")
    print(f"apply_mask={apply_mask}")
    print(f"include_load={args.include_load}")
    print(f"load_s={load_s:.6f}")
    print(f"warmup={args.warmup}")
    print(f"iterations={len(times)}")
    print(f"mean_s={statistics.fmean(times):.6f}")
    print(f"median_s={statistics.median(times):.6f}")
    print(f"min_s={min(times):.6f}")
    print(f"max_s={max(times):.6f}")


if __name__ == "__main__":
    main()
