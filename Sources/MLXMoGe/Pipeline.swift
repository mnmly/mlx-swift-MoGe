// High-level pipeline API: CGImage in, geometry out.
//
// Wraps `MoGeModel` so callers don't have to deal with manual NHWC tensor
// conversion or remember which `infer` flags to pass. Mirrors the shape of
// `DepthAnything3Pipeline` in mlx-swift-da3.

import CoreGraphics
import Foundation
import MLX

public struct MoGePrediction {
    public let outputs: [String: MLXArray]
    public let sourceHeight: Int
    public let sourceWidth: Int

    public init(outputs: [String: MLXArray], sourceHeight: Int, sourceWidth: Int) {
        self.outputs = outputs
        self.sourceHeight = sourceHeight
        self.sourceWidth = sourceWidth
    }

    public var points: MLXArray? { outputs["points"] }
    public var depth: MLXArray? { outputs["depth"] }
    public var normal: MLXArray? { outputs["normal"] }
    public var mask: MLXArray? { outputs["mask"] }
    public var intrinsics: MLXArray? { outputs["intrinsics"] }
}

public struct MoGePipeline {
    public let model: MoGeModel
    public let dtype: DType

    public init(model: MoGeModel, dtype: DType = .float16) {
        self.model = model
        self.dtype = dtype
    }

    /// Convert a `CGImage` to the row-major NHWC float32 `[0, 1]` tensor that
    /// `MoGeModel.infer` expects. Uses sRGB + premultiplied-last RGBA so the
    /// byte layout matches PIL + numpy on the Python side.
    public func preprocess(_ image: CGImage) -> MLXArray {
        cgImageToNHWC(image)
    }

    public func predict(
        _ input: MLXArray,
        numTokens: Int? = nil,
        resolutionLevel: Int = 9,
        forceProjection: Bool = true,
        applyMask: Bool = true,
        fovX: Float? = nil
    ) -> [String: MLXArray] {
        let outputs = model.infer(
            image: input,
            numTokens: numTokens,
            resolutionLevel: resolutionLevel,
            forceProjection: forceProjection,
            applyMask: applyMask,
            fovX: fovX
        )
        eval(Array(outputs.values))
        return outputs
    }

    public func callAsFunction(
        _ image: CGImage,
        numTokens: Int? = nil,
        resolutionLevel: Int = 9,
        forceProjection: Bool = true,
        applyMask: Bool = true,
        fovX: Float? = nil
    ) -> MoGePrediction {
        let input = preprocess(image)
        let outputs = predict(
            input,
            numTokens: numTokens,
            resolutionLevel: resolutionLevel,
            forceProjection: forceProjection,
            applyMask: applyMask,
            fovX: fovX
        )
        return MoGePrediction(
            outputs: outputs,
            sourceHeight: image.height,
            sourceWidth: image.width
        )
    }
}

public extension MoGePipeline {
    static func fromPretrained(
        _ path: String,
        dtype: DType = .float16
    ) throws -> MoGePipeline {
        let model = try MoGeModel.fromPretrained(path: path, dtype: dtype)
        return MoGePipeline(model: model, dtype: dtype)
    }

    static func fromPretrained(
        url: URL,
        dtype: DType = .float16
    ) throws -> MoGePipeline {
        try fromPretrained(url.path, dtype: dtype)
    }
}

public enum MLXMoGe {
    public static func fromPretrained(
        _ path: String,
        dtype: DType = .float16
    ) throws -> MoGePipeline {
        try MoGePipeline.fromPretrained(path, dtype: dtype)
    }
}

// MARK: - CGImage → NHWC

/// Decode a `CGImage` to a row-major NHWC float32 tensor in `[0, 1]`.
///
/// We deliberately avoid `NSBitmapImageRep.colorAt(x:y:)` — it goes through
/// per-pixel color management which is slow and introduces tiny noise that
/// MoGe's depth head amplifies into visible zigzag patterns in the output.
public func cgImageToNHWC(_ image: CGImage) -> MLXArray {
    let W = image.width
    let H = image.height
    let bytesPerRow = W * 4
    var rgba = [UInt8](repeating: 0, count: H * bytesPerRow)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue
        | CGBitmapInfo.byteOrder32Big.rawValue

    rgba.withUnsafeMutableBytes { buf in
        guard let ctx = CGContext(
            data: buf.baseAddress, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo
        ) else { return }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: W, height: H))
    }

    var floats = [Float](repeating: 0, count: H * W * 3)
    for i in 0..<(H * W) {
        floats[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
        floats[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
        floats[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
    }
    return MLXArray(floats, [H, W, 3])
}
