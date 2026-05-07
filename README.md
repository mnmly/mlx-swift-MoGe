# mlx-swift-MoGe

Swift / [mlx-swift](https://github.com/ml-explore/mlx-swift) port of
[MoGe-2](https://github.com/microsoft/MoGe) (Monocular Geometry Estimation) for
Apple Silicon. From a single RGB image, predicts a dense 3D point map, metric
depth, surface normals, a validity mask, and camera intrinsics.

Ported from the Python reference in
[mnmly/mlx-MoGe](https://github.com/mnmly/mlx-MoGe) and designed to consume the
same converted weights (`config.json` + `weights.safetensors`).

## Requirements

- macOS 14+
- Swift 5.9+ / Xcode 15+
- Apple Silicon (Metal is used via mlx-swift)

## Install

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mnmly/mlx-swift-MoGe", branch: "main"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MLXMoGe", package: "mlx-swift-MoGe"),
        ]
    ),
]
```

## Weights

This package does not ship weights. Convert a MoGe-2 PyTorch checkpoint once
using the Python port:

```bash
git clone https://github.com/mnmly/mlx-MoGe
cd mlx-MoGe
uv sync --extra convert

# float16 (default) — ~650 MB
mlx-moge-convert --model Ruicheng/moge-2-vitl-normal --output weights/

# float32 — ~1.3 GB, closer to PyTorch accuracy
mlx-moge-convert --model Ruicheng/moge-2-vitl-normal --output weights-fp32/ --float32
```

This produces a directory containing `config.json` and `weights.safetensors`
that `MoGeModel.fromPretrained(path:)` loads directly.

## Usage

### High-level pipeline (recommended)

`MoGePipeline` takes a `CGImage` in and gives you geometry out — preprocessing
to row-major NHWC `[0, 1]` is handled internally so byte layout matches the
Python reference (sRGB, premultiplied-last RGBA, no per-pixel color
management).

```swift
import CoreGraphics
import ImageIO
import MLX
import MLXMoGe

// 1. Load the model (defaults to float16).
let pipeline = try MoGePipeline.fromPretrained("/path/to/weights")

// 2. Decode any image to a CGImage.
let url = URL(fileURLWithPath: "photo.jpg")
let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!

// 3. Run the full pipeline. Defaults match `MoGeModel.infer`.
let prediction = pipeline(image)               // or: pipeline(image, resolutionLevel: 7)

let depth      = prediction.depth!             // (H, W)
let points     = prediction.points!            // (H, W, 3)
let normal     = prediction.normal!            // (H, W, 3)
let mask       = prediction.mask!              // (H, W) — 0/1
let intrinsics = prediction.intrinsics!        // (3, 3)
```

If you want to amortize preprocessing across many calls, drive the pipeline
in two steps:

```swift
let input = pipeline.preprocess(image)         // CGImage → NHWC float32
let outputs = pipeline.predict(input, resolutionLevel: 9)
```

### Lower-level: `MoGeModel.infer`

If you already have an `MLXArray` (e.g. from a custom decoder, a video frame,
or a batched `(B, H, W, 3)` tensor), call `MoGeModel.infer` directly:

```swift
import MLX
import MLXMoGe

let model = try MoGeModel.fromPretrained(path: "/path/to/weights")

// image: (H, W, 3) or (B, H, W, 3) NHWC float in [0, 1].
let result = model.infer(
    image: image,
    resolutionLevel: 9,   // 0–9, higher = finer detail
    forceProjection: true,
    applyMask: true
)
MLX.eval(Array(result.values))
```

The free function `cgImageToNHWC(_:)` exposes the same `CGImage → NHWC`
conversion the pipeline uses, in case you want preprocessing without the
pipeline wrapper.

### `MoGeModel.infer(...)` parameters

| Parameter         | Type       | Default | Description                                           |
|-------------------|------------|---------|-------------------------------------------------------|
| `image`           | `MLXArray` | —       | `(H, W, 3)` or `(B, H, W, 3)` NHWC float in `[0, 1]`  |
| `numTokens`       | `Int?`     | `nil`   | Override ViT token count directly                     |
| `resolutionLevel` | `Int`      | `9`     | `0…9`, higher = more ViT tokens = more detail         |
| `forceProjection` | `Bool`     | `true`  | Recompute points from depth + recovered intrinsics    |
| `applyMask`       | `Bool`     | `true`  | Set invalid pixels in points/depth/normal to `+inf`/0 |
| `fovX`            | `Float?`   | `nil`   | Known horizontal FOV in radians (skips focal recovery)|

The signature mirrors Python `MoGeModel.infer` in
[mlx-MoGe](https://github.com/mnmly/mlx-MoGe), including the
Levenberg-Marquardt focal/shift recovery (reimplemented in pure Swift here
since `scipy.optimize.least_squares` is unavailable).

### Loading at `float32`

```swift
let model = try MoGeModel.fromPretrained(path: "weights-fp32", dtype: .float32)
```

Point the path at a weights directory that was converted with `--float32`.
Mixing a `float32` dtype argument with `float16` weight files (or vice versa)
will load fine but waste memory, so keep separate directories per precision.

## Package layout

```
Sources/MLXMoGe/
  ImageUtils.swift        # bilinear_resize, pad_replicate, conv_transpose2d
  GeometryUtils.swift     # UV grid, intrinsics builder, depth → points
  MoGeModules.swift       # ResidualConvBlock, Resampler, ConvStack, ScaleHead
  DINOv2.swift            # ViT-L/14 backbone with intermediate extraction
  MoGeModel.swift         # DINOv2Encoder + MoGeModel forward pass
  WeightLoading.swift     # config.json + safetensors → MoGeModel
  MoGeInference.swift     # infer() wrapper + pure-Swift LM focal/shift solver
Tests/MLXMoGeTests/
  ShapeTests.swift        # per-module shape & sanity checks
  InferenceTests.swift    # LM self-consistency + weight-key remap
```

## Benchmarks

Head-to-head against the Torch reference in
[microsoft/MoGe](https://github.com/microsoft/MoGe), same model
(`Ruicheng/moge-2-vitl-normal`), same input
(`example_images/01_HouseIndoor.jpg`, 1500×1000), fp16, batch=1,
`resolution_level=9`, 2 warmup + 10 measured iterations.

Hardware: Apple Silicon, macOS 14+. Torch backend is MPS.

| Variant         | Backend     | Median   | Mean     | Min      | Max      |
|-----------------|-------------|----------|----------|----------|----------|
| inference-only  | Torch / MPS | 0.648 s  | 0.652 s  | 0.641 s  | 0.671 s  |
| inference-only  | **mlx-swift** | **0.320 s** | 0.319 s | 0.316 s | 0.324 s |
| end-to-end\*    | Torch / MPS | 0.662 s  | 0.661 s  | 0.650 s  | 0.666 s  |
| end-to-end\*    | **mlx-swift** | **0.332 s** | 0.332 s | 0.326 s | 0.338 s |
| cold model load | Torch / MPS | —        | 2.89 s   | —        | —        |
| cold model load | **mlx-swift** | —      | **0.25 s** | —      | —        |

\* End-to-end includes file decode + tensor upload.

mlx-swift is ~**2.0× faster** on inference and ~**12× faster** to cold-load
the weights. Reproduce with the scripts under
[`Benchmarks/`](Benchmarks/README.md):

```bash
# Swift
xcodebuild -scheme moge-bench -destination 'platform=macOS' \
  -configuration release -derivedDataPath .xcdd build
.xcdd/Build/Products/release/moge-bench \
    --weights /path/to/weights --input photo.jpg

# Python (run from microsoft/MoGe checkout)
python Benchmarks/torch_moge_bench.py --input photo.jpg --device mps
```

## Tests

Running the test suite requires `xcodebuild` (not `swift test`) because
mlx-swift's Metal library bundle is only resolved by Xcode's test runner:

```bash
xcodebuild -scheme mlx-swift-moge-Package -destination 'platform=macOS' test
```

Shape tests run with random weights and do not require a checkpoint. The
end-to-end parity test (`ParityFixtureTests`) compares Swift `infer` outputs
against a Torch reference fixture and is skipped unless both the fixture and
weights are available:

```bash
# 1. Generate the fixture once (needs the python/MoGe checkout):
cd /path/to/python/MoGe
uv run --with safetensors python /path/to/mlx-swift-MoGe/Scripts/generate_parity_fixture.py \
    --image example_images/01_HouseIndoor.jpg \
    --out /path/to/mlx-swift-MoGe/Tests/Fixtures

# 2. Run with MOGE_WEIGHTS pointing at the converted mlx weights. The
#    TEST_RUNNER_ prefix is required to forward the env var into the xctest
#    process — `xcodebuild test` does not inherit shell env by default.
cd /path/to/mlx-swift-MoGe
TEST_RUNNER_MOGE_WEIGHTS=/path/to/weights \
    xcodebuild -scheme mlx-swift-moge-Package -destination 'platform=macOS' \
    -derivedDataPath .xcdd test -only-testing:MLXMoGeTests/ParityFixtureTests
```

The test reports per-output `maxAbs` and `meanAbs`. Tolerances are loose on
the per-element `maxAbs` (a few edge pixels diverge after bilinear upsample
under fp16) but tight on `meanAbs`, which is what catches real regressions.

## Acknowledgements

- [MoGe](https://github.com/microsoft/MoGe) — original PyTorch implementation
  by Ruicheng Wang et al. at Microsoft Research.
- [mlx-MoGe](https://github.com/mnmly/mlx-MoGe) — the Python MLX port this
  Swift package mirrors file-for-file.
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — MLX bindings for
  Swift by Apple.
- [DINOv2](https://github.com/facebookresearch/dinov2) — ViT backbone by Meta.

## License

MIT — see [LICENSE](LICENSE) for full text, including third-party notices for
MoGe, DINOv2, mlx-swift, and swift-numerics.
