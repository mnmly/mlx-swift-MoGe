// Example: Full inference pipeline with colored point cloud PLY export
//
// Demonstrates the high-level `MoGePipeline` API:
//   1. Decode an image to `CGImage` (any decoder works).
//   2. Use `pipeline.preprocess` once so we can also reuse the same NHWC
//      tensor for per-pixel color extraction below.
//   3. Run `pipeline.predict` for `points`, `depth`, `mask`.
//   4. Filter invalid pixels and write a colored PLY.
import AppKit
import CoreGraphics
import ImageIO
import MLX
import MLXMoGe

func loadCGImage(path: String) -> CGImage {
    let url = URL(fileURLWithPath: path) as CFURL
    if let source = CGImageSourceCreateWithURL(url, nil),
       CGImageSourceGetCount(source) > 0,
       let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    {
        return image
    }
    if let nsImage = NSImage(contentsOf: url as URL),
       let image = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
    {
        return image
    }
    fatalError("Could not load image: \(path)")
}

func runExample(imagePath: String, weightsPath: String, outputPlyPath: String) throws {
    print("Loading model from \(weightsPath)")
    let pipeline = try MoGePipeline.fromPretrained(weightsPath)

    print("Loading image from \(imagePath)")
    let cgImage = loadCGImage(path: imagePath)

    // Two-step pipeline use: preprocess once so we can reuse the NHWC tensor
    // for both inference and per-pixel color extraction.
    let imageNHWC = pipeline.preprocess(cgImage)
    print(imageNHWC.shape)

    print("Running inference...")
    let outputs = pipeline.predict(
        imageNHWC,
        resolutionLevel: 9,
        forceProjection: true,
        applyMask: true
    )

    let points = outputs["points"]!  // (H, W, 3)
    let depth = outputs["depth"]!    // (H, W)
    let mask = outputs["mask"]!      // (H, W) — 1.0 where valid

    // Pull everything to CPU in row-major order — same layout Python uses,
    // so points[i] aligns with colors[i] pixel-for-pixel.
    let pointData: [Float] = points.asArray(Float.self)
    let depthData: [Float] = depth.asArray(Float.self)
    let maskData: [Float] = mask.asArray(Float.self)
    let colorData: [Float] = (imageNHWC * 255.0).asType(.float32).asArray(Float.self)

    let pixelCount = depthData.count
    var vertices: [PLYVertex] = []
    vertices.reserveCapacity(pixelCount)

    for i in 0..<pixelCount {
        guard maskData[i] > 0.5 else { continue }
        guard depthData[i].isFinite else { continue }
        let px = pointData[i * 3 + 0]
        let py = pointData[i * 3 + 1]
        let pz = pointData[i * 3 + 2]
        guard px.isFinite, py.isFinite, pz.isFinite else { continue }

        let r = UInt8(min(max(colorData[i * 3 + 0], 0), 255))
        let g = UInt8(min(max(colorData[i * 3 + 1], 0), 255))
        let b = UInt8(min(max(colorData[i * 3 + 2], 0), 255))

        vertices.append(PLYVertex(x: px, y: py, z: pz, r: r, g: g, b: b))
    }

    print("Exporting \(vertices.count) points to PLY...")
    try exportPLY(to: outputPlyPath, vertices: vertices, faces: nil, binary: true)
    print("Done! Exported to \(outputPlyPath)")
}

guard CommandLine.arguments.count >= 4 else {
    print("Usage: PlyExportExample <image_path> <weights_path> <output_ply_path>")
    print("Example: PlyExportExample photo.jpg ./weights ./output.ply")
    exit(1)
}
let imagePath = CommandLine.arguments[1]
let weightsPath = CommandLine.arguments[2]
let outputPlyPath = CommandLine.arguments[3]
do {
    try runExample(imagePath: imagePath, weightsPath: weightsPath, outputPlyPath: outputPlyPath)
} catch {
    print("Error: \(error)")
    exit(1)
}
