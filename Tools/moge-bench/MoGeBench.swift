import AppKit
import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXMoGe

@main
struct MoGeBench: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "moge-bench",
        abstract: "Benchmark MLX Swift MoGe-2 inference."
    )

    @Option(name: .shortAndLong, help: "Path to converted weights directory (config.json + weights.safetensors).")
    var weights: String

    @Option(name: .shortAndLong, help: "Input image path.")
    var input: String

    @Option(name: .long, help: "Weight dtype: float16 or float32.")
    var dtype: String = "float16"

    @Option(name: .long, help: "Resolution level [0, 9].")
    var resolutionLevel: Int = 9

    @Option(name: .long, help: "Override token count (optional).")
    var numTokens: Int? = nil

    @Flag(name: .long, inversion: .prefixedNo, help: "Recompute points from depth + intrinsics.")
    var forceProjection: Bool = true

    @Flag(name: .long, inversion: .prefixedNo, help: "Mask invalid regions to +infinity.")
    var applyMask: Bool = true

    @Option(name: .long, help: "Warmup iterations.")
    var warmup: Int = 2

    @Option(name: .long, help: "Measured iterations.")
    var iterations: Int = 10

    @Flag(name: .long, help: "Include image load (file -> NHWC MLXArray) in measured time.")
    var includeLoad: Bool = false

    func run() throws {
        let targetDtype: DType = (dtype == "float32") ? .float32 : .float16

        let loadStart = CFAbsoluteTimeGetCurrent()
        let model = try MoGeModel.fromPretrained(path: weights, dtype: targetDtype)
        let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart

        guard let preloaded = loadImageAsNHWC(path: input) else {
            throw ValidationError("Could not load image at \(input)")
        }
        let (preImage, H, W) = preloaded
        eval(preImage)

        let runOnce: () -> Void = { [self] in
            let img: MLXArray
            if includeLoad {
                guard let (loaded, _, _) = loadImageAsNHWC(path: input) else {
                    fatalError("Could not load image at \(input)")
                }
                img = loaded
            } else {
                img = preImage
            }
            let out = model.infer(
                image: img,
                numTokens: numTokens,
                resolutionLevel: resolutionLevel,
                forceProjection: forceProjection,
                applyMask: applyMask
            )
            // Force materialization on this stream before timing stops.
            eval(Array(out.values))
        }

        for _ in 0..<max(0, warmup) {
            runOnce()
        }

        var times: [Double] = []
        for _ in 0..<max(1, iterations) {
            let start = CFAbsoluteTimeGetCurrent()
            runOnce()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        let sorted = times.sorted()
        let median = sorted[sorted.count / 2]
        let minTime = sorted.first ?? 0
        let maxTime = sorted.last ?? 0

        print("backend=mlx-swift")
        print("model=Ruicheng/moge-2-vitl-normal")
        print("dtype=\(dtype)")
        print("input=\(input)")
        print("source_size=\(W)x\(H)")
        print("resolution_level=\(resolutionLevel)")
        print("num_tokens=\(numTokens.map { String($0) } ?? "auto")")
        print("force_projection=\(forceProjection)")
        print("apply_mask=\(applyMask)")
        print("include_load=\(includeLoad)")
        print(String(format: "load_s=%.6f", loadSeconds))
        print("warmup=\(warmup)")
        print("iterations=\(times.count)")
        print(String(format: "mean_s=%.6f", mean))
        print(String(format: "median_s=%.6f", median))
        print(String(format: "min_s=%.6f", minTime))
        print(String(format: "max_s=%.6f", maxTime))
    }
}

// Match the row-major NHWC float32 [0, 1] layout from
// `Examples/PlyExportExample/Sources/PlyExportExample/main.swift`.
private func loadImageAsNHWC(path: String) -> (MLXArray, Int, Int)? {
    let url = URL(fileURLWithPath: path)
    let cgImage: CGImage? = {
        if let provider = CGDataProvider(url: url as CFURL),
           let png = CGImage(
               pngDataProviderSource: provider,
               decode: nil, shouldInterpolate: false, intent: .defaultIntent
           )
        {
            return png
        }
        return NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }()

    guard let cg = cgImage else { return nil }

    let W = cg.width
    let H = cg.height
    let bytesPerRow = W * 4
    var rgba = [UInt8](repeating: 0, count: H * bytesPerRow)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue
        | CGBitmapInfo.byteOrder32Big.rawValue

    guard let ctx = rgba.withUnsafeMutableBytes({ buf -> CGContext? in
        CGContext(
            data: buf.baseAddress, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo
        )
    }) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))

    var floats = [Float](repeating: 0, count: H * W * 3)
    for i in 0..<(H * W) {
        floats[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
        floats[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
        floats[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
    }
    return (MLXArray(floats, [H, W, 3]), H, W)
}
