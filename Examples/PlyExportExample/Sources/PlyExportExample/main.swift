// Example: Full inference pipeline with colored point cloud PLY export
//
// This example demonstrates:
// 1. Loading an image (NSImage on macOS)
// 2. Running MoGe-2 inference
// 3. Exporting a colored point cloud to PLY format
import MLX
import MLXMoGe
import AppKit
import CoreGraphics
// MARK: - Image Loading Utility
//
// Load an image as a row-major NHWC float32 tensor in [0, 1], matching what
// PIL + numpy produce in the Python reference. We intentionally avoid
// `NSBitmapImageRep.colorAt(x:y:)` — it returns per-pixel `NSColor` objects
// and goes through color management, which is slow *and* introduces tiny
// per-pixel noise that MoGe's depth head amplifies into visible zigzag.
func loadImageAsNHWC(path: String) -> (MLXArray, Int, Int) {
    guard
        let dataProvider = CGDataProvider(url: URL(fileURLWithPath: path) as CFURL),
        let cgImage = CGImage(
            pngDataProviderSource: dataProvider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) ?? NSImage(contentsOfFile: path)?
            .cgImage(forProposedRect: nil, context: nil, hints: nil)
    else {
        fatalError("Could not load image: \(path)")
    }

    let W = cgImage.width
    let H = cgImage.height

    // Draw into a tightly-packed RGBA8 buffer in sRGB — same bytes PIL gives.
    let bytesPerPixel = 4
    let bytesPerRow = W * bytesPerPixel
    var rgba = [UInt8](repeating: 0, count: H * bytesPerRow)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue
        | CGBitmapInfo.byteOrder32Big.rawValue

    guard let ctx = rgba.withUnsafeMutableBytes({ buf -> CGContext? in
        CGContext(
            data: buf.baseAddress,
            width: W, height: H,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        )
    }) else {
        fatalError("Could not create CGContext")
    }
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: W, height: H))

    // Strip alpha and convert to float32 [0, 1], row-major (top-left origin).
    var floats = [Float](repeating: 0, count: H * W * 3)
    for i in 0..<(H * W) {
        floats[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
        floats[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
        floats[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
    }

    return (MLXArray(floats, [H, W, 3]), H, W)
}
// MARK: - Main Example
func runExample(imagePath: String, weightsPath: String, outputPlyPath: String) throws {
    print("Loading model from \(weightsPath)")
    
    // Load the MoGe model
    let model = try MoGeModel.fromPretrained(path: weightsPath)
    
    print("Loading image from \(imagePath)")

    let (image, _, _) = loadImageAsNHWC(path: imagePath)
    print(image.shape)
    print("Running inference...")

    // Run full inference. Keep applyMask on so invalid pixels are marked +inf,
    // then we filter them out before writing the PLY — mirroring the Python
    // reference in ../../python/mlx-MoGe/examples/estimate_depth.py.
    let result = model.infer(
        image: image,
        resolutionLevel: 9,
        forceProjection: true,
        applyMask: true
    )

    let points = result["points"]!  // (H, W, 3)
    let depth = result["depth"]!    // (H, W)
    let mask = result["mask"]!      // (H, W) — 1.0 where valid
    MLX.eval(points, depth, mask)

    // Pull everything to CPU in row-major order — same layout Python uses,
    // so points[i] aligns with colors[i] pixel-for-pixel.
    let pointData: [Float] = points.asArray(Float.self)
    let depthData: [Float] = depth.asArray(Float.self)
    let maskData: [Float] = mask.asArray(Float.self)
    let colorData: [Float] = (image * 255.0).asType(.float32).asArray(Float.self)

    let pixelCount = depthData.count
    var vertices: [PLYVertex] = []
    vertices.reserveCapacity(pixelCount)

    for i in 0..<pixelCount {
        // valid = mask & isFinite(depth) & isFinite(point)
        guard maskData[i] > 0.5 else { continue }
        guard depthData[i].isFinite else { continue }
        let px = pointData[i * 3 + 0]
        let py = pointData[i * 3 + 1]
        let pz = pointData[i * 3 + 2]
        guard px.isFinite, py.isFinite, pz.isFinite else { continue }

        let r = UInt8(min(max(colorData[i * 3 + 0], 0), 255))
        let g = UInt8(min(max(colorData[i * 3 + 1], 0), 255))
        let b = UInt8(min(max(colorData[i * 3 + 2], 0), 255))

        vertices.append(
            PLYVertex(x: px, y: py, z: pz, r: r, g: g, b: b)
        )
    }

    print("Exporting \(vertices.count) points to PLY...")

    try exportPLY(to: outputPlyPath, vertices: vertices, faces: nil, binary: true)

    print("Done! Exported to \(outputPlyPath)")
}
// MARK: - Entry Point
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
