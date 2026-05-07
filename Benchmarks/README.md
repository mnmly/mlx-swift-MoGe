# MoGe-2 Benchmarks (Python vs mlx-swift)

Same protocol as the `mlx-swift-da3` and `mlx-swift-sapiens2` packages: warmup
+ measured iterations, identical model variant (`Ruicheng/moge-2-vitl-normal`),
fp16 by default, batch=1.

Two timings are reported per backend:

- **inference-only** (default): image is preloaded once; the timed loop runs
  `model.infer` only.
- **end-to-end** (`--include-load`): the timed loop also reloads the image
  from disk and uploads it to the device.

## Python (Torch, MPS)

```bash
cd ../../python/MoGe
uv sync                                # install deps for the python MoGe repo
uv run python ../../swift/mlx-swift-MoGe/Benchmarks/torch_moge_bench.py \
    --input example_images/01_HouseIndoor.jpg \
    --device mps \
    --warmup 2 --iterations 10
```

Add `--include-load` for end-to-end. `--no-fp16` runs the fp32 path.

## Swift (mlx-swift)

Build with `xcodebuild` so MLX gets a Metal-capable toolchain (using
`swift run` skips Metal codesigning and the GPU path won't initialize):

```bash
xcodebuild -scheme moge-bench -destination 'platform=macOS' \
  -configuration release -derivedDataPath .xcdd build

.xcdd/Build/Products/release/moge-bench \
    --weights /path/to/weights \
    --input ../../python/MoGe/example_images/01_HouseIndoor.jpg \
    --warmup 2 --iterations 10
```

Pass `--include-load` for end-to-end, `--dtype float32` for fp32. The weights
directory is the converted `config.json` + `weights.safetensors` produced by
`mlx-moge-convert` (see the top-level README).

## Notes

- Both scripts call `infer` with `resolution_level=9`, `force_projection=true`,
  `apply_mask=true` — same defaults the Swift example uses.
- Use the same input image for both runs; resolution affects token count.
- MPS times include the device sync (`torch.mps.synchronize`); Swift times
  include `MLX.eval` on the returned arrays.
