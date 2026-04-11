// Image processing utilities for MoGe-2 Swift port
// bilinear_resize and pad_replicate for NHWC tensors

import MLX
import MLXNN

/// Replicate (edge) padding for NHWC tensors.
///
/// - Parameters:
///   - x: Input tensor (B, H, W, C)
///   - pad: Number of pixels to pad on each side of the spatial dims
/// - Returns:
///   Padded tensor (B, H+2*pad, W+2*pad, C)
public func padReplicate(_ x: MLXArray, _ pad: Int) -> MLXArray {
    if pad == 0 { return x }
    // Pad only spatial axes (1 and 2), leave batch and channel unchanged.
    let widths: [IntOrPair] = [
        IntOrPair(0), IntOrPair(pad), IntOrPair(pad), IntOrPair(0)
    ]
    return padded(x, widths: widths, mode: .edge)
}

/// Transposed convolution (2x upsampling) implemented via `convGeneral` with
/// `inputDilation=stride`. Weights must be pre-flipped spatially (as done by
/// `convert.py` for ConvTranspose2d keys) and laid out `(outC, kH, kW, inC)`.
public func convTranspose2d(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray?,
    stride: Int = 2
) -> MLXArray {
    let kernelSize = weight.dim(1)
    let padding = kernelSize - 1 // produces output of size input*stride
    var out = convGeneral(
        x, weight,
        strides: .init(1),
        padding: .init(padding),
        inputDilation: .init(stride)
    )
    if let bias { out = out + bias }
    return out
}

/// Bilinear interpolation resize for NHWC tensors. Matches PyTorch's
/// `F.interpolate(..., mode='bilinear', align_corners=False)`.
///
/// - Parameters:
///   - x: Input (B, H, W, C)
///   - targetH: Target height
///   - targetW: Target width
/// - Returns:
///   Resized tensor (B, targetH, targetW, C)
public func bilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let H = x.dim(1)
    let W = x.dim(2)
    if H == targetH && W == targetW { return x }

    let sh = Float(targetH) / Float(H)
    let sw = Float(targetW) / Float(W)
    let up = Upsample(scaleFactor: .array([sh, sw]), mode: .linear(alignCorners: false))
    return up(x)
}
