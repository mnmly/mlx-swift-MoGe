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

```swift
import MLX
import MLXMoGe
import AppKit   // or UIKit

// 1. Load the model (defaults to float16).
let model = try MoGeModel.fromPretrained(path: "/path/to/weights")

// 2. Load an image as NHWC float in [0, 1]. Any decoder works; here's a
//    minimal NSImage → MLXArray conversion for macOS.
func imageToMLX(_ url: URL) -> MLXArray {
    let image = NSImage(contentsOf: url)!
    let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
    let H = rep.pixelsHigh
    let W = rep.pixelsWide
    var floats = [Float](repeating: 0, count: H * W * 3)
    for y in 0..<H {
        for x in 0..<W {
            let c = rep.colorAt(x: x, y: y)!
            let i = (y * W + x) * 3
            floats[i + 0] = Float(c.redComponent)
            floats[i + 1] = Float(c.greenComponent)
            floats[i + 2] = Float(c.blueComponent)
        }
    }
    return MLXArray(floats, [H, W, 3])
}

let image = imageToMLX(URL(fileURLWithPath: "photo.jpg"))

// 3. Run inference. Accepts (H, W, 3) or (B, H, W, 3).
let result = model.infer(
    image: image,
    resolutionLevel: 9,   // 0–9, higher = finer detail
    forceProjection: true,
    applyMask: true
)

// 4. Pull outputs out of the result dictionary.
let depth      = result["depth"]!        // (H, W)          Float
let points     = result["points"]!       // (H, W, 3)       Float
let normal     = result["normal"]!       // (H, W, 3)       Float
let mask       = result["mask"]!         // (H, W)          0/1
let intrinsics = result["intrinsics"]!   // (3, 3)          Float

MLX.eval(depth, points, normal, mask, intrinsics)
```

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

## Tests

Running the test suite requires `xcodebuild` (not `swift test`) because
mlx-swift's Metal library bundle is only resolved by Xcode's test runner:

```bash
xcodebuild -scheme mlx-swift-moge -destination 'platform=macOS' test
```

All 23 tests should pass. Shape tests run with random weights and do not
require a checkpoint.

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
