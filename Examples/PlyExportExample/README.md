# PlyExportExample

This example demonstrates how to use the MoGe-2 Swift library for monocular geometry estimation and export results to PLY format.

## Features

- Load images from disk (NSImage on macOS)
- Run full MoGe-2 inference pipeline
- Export depth maps to PLY format
- Support for both binary and ASCII PLY output

## Prerequisites

1. A converted MoGe-2 weights directory with `config.json` and `weights.safetensors`
2. An input image (JPG, PNG, etc.)

## Building

```bash
cd Examples/PlyExportExample
swift build
```

## Running

```bash
swift run PlyExportExample <image_path> <weights_path> <output_ply_path>
```

### Example:

```bash
swift run PlyExportExample \
  ~/Pictures/photo.jpg \
  ./weights \
  ./output.ply
```

## API Usage

```swift
import MLXMoGe

// Load model
let model = try MoGeModel.fromPretrained(path: "path/to/weights")

// Load image (as MLXArray in NHWC format, [0,1] range)
let image: MLXArray = ...

// Run inference
let result = model.infer(
    image: image,
    resolutionLevel: 9,
    forceProjection: true,
    applyMask: true
)

// Extract results
let depth = result["depth"]!
let points = result["points"]!
let intrinsics = result["intrinsics"]!

// Export to PLY
try exportDepthMapToPLY(
    path: "output.ply",
    depth: depth,
    intrinsics: intrinsics,
    height: image.dim(0),
    width: image.dim(1)
)
```

## Available Export Functions

- `exportDepthMapToPLY` - From depth map tensor
- `exportPointCloudToPLY` - From point cloud tensor  
- `exportPointCloudWithColorsToPLY` - With color information
- `exportMeshToPLY` - Triangle mesh
- `exportMeshWithNormalsToPLY` - Mesh with vertex normals

## Output Format

The PLY files are exported in binary little-endian format by default, containing:
- Vertex positions (x, y, z)
- Optional: vertex normals (nx, ny, nz)
- Optional: vertex colors (r, g, b)
- Triangular faces (3 indices per face)

