// Inference wrapper for MoGe-2. Mirrors `MoGeModel.infer` in the Python
// reference (`mlx_moge/model/moge.py`), including the focal/shift recovery
// that Python delegates to `scipy.optimize.least_squares` (Levenberg-Marquardt).
//
// Post-processing is performed on the CPU with `[Float]` arrays because the
// optimization loop is inherently sequential and per-batch. Results are
// re-materialized as MLXArrays before returning.

import Foundation
import MLX

extension MoGeModel {

    /// Full inference with post-processing.
    ///
    /// - Parameters:
    ///   - image: `(H, W, 3)` or `(B, H, W, 3)` NHWC float in `[0, 1]`.
    ///   - numTokens: Override token count. Defaults to the value derived
    ///     from `resolutionLevel` and `numTokensRange`.
    ///   - resolutionLevel: Integer in `[0, 9]` selecting a point inside
    ///     `numTokensRange`.
    ///   - forceProjection: Recompute points from depth + intrinsics.
    ///   - applyMask: Set invalid regions to `+infinity`.
    ///   - fovX: Known horizontal field of view in radians (optional).
    /// - Returns: Dictionary with `points`, `depth`, `intrinsics`, optional
    ///   `mask` and `normal`.
    public func infer(
        image: MLXArray,
        numTokens: Int? = nil,
        resolutionLevel: Int = 9,
        forceProjection: Bool = true,
        applyMask: Bool = true,
        fovX: Float? = nil
    ) -> [String: MLXArray] {
        // Accept (H, W, 3) by prepending a batch dim.
        let omitBatch = image.ndim == 3
        let batched = omitBatch ? image.expandedDimensions(axis: 0) : image

        let B = batched.dim(0)
        let imgH = batched.dim(1)
        let imgW = batched.dim(2)
        let aspectRatio = Float(imgW) / Float(imgH)

        // Token count from resolution_level.
        let resolvedNumTokens: Int
        if let n = numTokens {
            resolvedNumTokens = n
        } else {
            let minT = numTokensRange[0]
            let maxT = numTokensRange[1]
            resolvedNumTokens = minT + Int(Float(resolutionLevel) / 9.0 * Float(maxT - minT))
        }

        // Forward pass.
        let output = self(batched, resolvedNumTokens)
        MLX.eval(Array(output.values))

        // Materialize points and mask on the CPU for optimization.
        let pointsArr = output["points"]!.asType(.float32)
        MLX.eval(pointsArr)
        let pointsFlat: [Float] = pointsArr.asArray(Float.self)

        var maskFlat: [Bool]? = nil
        if let m = output["mask"] {
            let mf: [Float] = m.asType(.float32).asArray(Float.self)
            maskFlat = mf.map { $0 > 0.5 }
        }

        // Recover per-batch focal and shift.
        var focals = [Float](repeating: 1, count: B)
        var shifts = [Float](repeating: 0, count: B)
        let stride = imgH * imgW * 3
        let maskStride = imgH * imgW

        for b in 0..<B {
            let ptsSlice = Array(pointsFlat[(b * stride)..<((b + 1) * stride)])
            let maskSlice: [Bool]? = maskFlat.map {
                Array($0[(b * maskStride)..<((b + 1) * maskStride)])
            }

            if let fx = fovX {
                let knownFocal =
                    aspectRatio / (1 + aspectRatio * aspectRatio).squareRoot()
                    / tanf(fx / 2)
                shifts[b] = Self.solveOptimalShift(
                    points: ptsSlice, mask: maskSlice,
                    H: imgH, W: imgW, aspectRatio: aspectRatio,
                    focal: knownFocal
                )
                focals[b] = knownFocal
            } else {
                let (f, s) = Self.solveOptimalFocalShift(
                    points: ptsSlice, mask: maskSlice,
                    H: imgH, W: imgW, aspectRatio: aspectRatio
                )
                focals[b] = f
                shifts[b] = s
            }
        }

        // Build depth: points_z + shift (broadcast over HxW).
        // Compose directly on CPU, then wrap as MLXArray.
        var depthFlat = [Float](repeating: 0, count: B * imgH * imgW)
        for b in 0..<B {
            let s = shifts[b]
            for i in 0..<(imgH * imgW) {
                let pz = pointsFlat[b * stride + i * 3 + 2]
                depthFlat[b * imgH * imgW + i] = pz + s
            }
        }

        // Clamp mask by positive depth.
        if var m = maskFlat {
            for i in 0..<m.count {
                if depthFlat[i] <= 0 { m[i] = false }
            }
            maskFlat = m
        }

        // Intrinsics.
        var fxArr = [Float](repeating: 0, count: B)
        var fyArr = [Float](repeating: 0, count: B)
        let sqrtTerm = (1 + aspectRatio * aspectRatio).squareRoot()
        for b in 0..<B {
            fxArr[b] = focals[b] / 2 * sqrtTerm / aspectRatio
            fyArr[b] = focals[b] / 2 * sqrtTerm
        }
        let cxArr = [Float](repeating: 0.5, count: B)
        let cyArr = [Float](repeating: 0.5, count: B)
        let intrinsicsMLX = intrinsicsFromFocalCenter(
            fx: fxArr, fy: fyArr, cx: cxArr, cy: cyArr)

        // Depth and (optionally) reprojected points as MLXArrays.
        var depthMLX = MLXArray(depthFlat, [B, imgH, imgW])
        var pointsMLX: MLXArray
        if forceProjection {
            pointsMLX = depthMapToPointMap(
                depth: depthMLX, intrinsics: intrinsicsMLX,
                height: imgH, width: imgW)
        } else {
            pointsMLX = MLXArray(pointsFlat, [B, imgH, imgW, 3])
            // Replace z channel with shifted depth.
            pointsMLX = pointsMLX.at[0..., 0..., 0..., 2...].add(
                depthMLX.expandedDimensions(axis: -1)
                - pointsMLX[0..., 0..., 0..., 2...])
        }

        // Metric scale, if predicted.
        if let scaleArr = output["metric_scale"] {
            let scale = scaleArr.asType(.float32).reshaped([B, 1, 1, 1])
            pointsMLX = pointsMLX * scale
            depthMLX = depthMLX * scale.squeezed(axis: -1)
        }

        // Apply mask → +inf on invalid pixels.
        if applyMask, let m = maskFlat {
            let inf = Float.infinity
            var pf: [Float] = pointsMLX.asArray(Float.self)
            var df: [Float] = depthMLX.asArray(Float.self)
            for b in 0..<B {
                for i in 0..<(imgH * imgW) {
                    if !m[b * imgH * imgW + i] {
                        pf[(b * imgH * imgW + i) * 3 + 0] = inf
                        pf[(b * imgH * imgW + i) * 3 + 1] = inf
                        pf[(b * imgH * imgW + i) * 3 + 2] = inf
                        df[b * imgH * imgW + i] = inf
                    }
                }
            }
            pointsMLX = MLXArray(pf, [B, imgH, imgW, 3])
            depthMLX = MLXArray(df, [B, imgH, imgW])
        }

        var result: [String: MLXArray] = [:]
        result["points"] = pointsMLX
        result["depth"] = depthMLX
        result["intrinsics"] = intrinsicsMLX

        if let m = maskFlat {
            let u8 = m.map { $0 ? Float(1) : Float(0) }
            result["mask"] = MLXArray(u8, [B, imgH, imgW])
        }
        if let normal = output["normal"] {
            var n = normal.asType(.float32)
            if applyMask, let m = maskFlat {
                var nf: [Float] = n.asArray(Float.self)
                for b in 0..<B {
                    for i in 0..<(imgH * imgW) {
                        if !m[b * imgH * imgW + i] {
                            nf[(b * imgH * imgW + i) * 3 + 0] = 0
                            nf[(b * imgH * imgW + i) * 3 + 1] = 0
                            nf[(b * imgH * imgW + i) * 3 + 2] = 0
                        }
                    }
                }
                n = MLXArray(nf, [B, imgH, imgW, 3])
            }
            result["normal"] = n
        }

        if omitBatch {
            for (k, v) in result {
                result[k] = v[0]
            }
        }
        return result
    }

    // MARK: - Focal/shift recovery (Levenberg-Marquardt)

    /// Collect UV and XYZ correspondences for a single sample, downsampled to
    /// roughly `64×64` and filtered by the validity mask.
    private static func gatherUVXYZ(
        points: [Float], mask: [Bool]?,
        H: Int, W: Int, aspectRatio: Float,
        downsample: Int = 64
    ) -> (uv: [Float], xyz: [Float]) {
        let spanX = aspectRatio / (1 + aspectRatio * aspectRatio).squareRoot()
        let spanY = 1.0 / (1 + aspectRatio * aspectRatio).squareRoot()

        // linspace endpoints inclusive (matches numpy).
        func linspace(_ a: Float, _ b: Float, _ n: Int) -> [Float] {
            if n <= 1 { return [a] }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n {
                out[i] = a + (b - a) * Float(i) / Float(n - 1)
            }
            return out
        }
        let us = linspace(-spanX, spanX, W)
        let vs = linspace(-spanY, spanY, H)

        let stepH = max(1, H / downsample)
        let stepW = max(1, W / downsample)

        var uv: [Float] = []
        var xyz: [Float] = []
        uv.reserveCapacity((H / stepH) * (W / stepW) * 2)
        xyz.reserveCapacity((H / stepH) * (W / stepW) * 3)

        var y = 0
        while y < H {
            var x = 0
            while x < W {
                let idx = y * W + x
                let valid = mask.map { $0[idx] } ?? true
                if valid {
                    uv.append(us[x])
                    uv.append(vs[y])
                    xyz.append(points[idx * 3 + 0])
                    xyz.append(points[idx * 3 + 1])
                    xyz.append(points[idx * 3 + 2])
                }
                x += stepW
            }
            y += stepH
        }
        return (uv, xyz)
    }

    /// Solve `min_{focal, shift} Σ ||focal * xy / (z + shift) - uv||²`.
    /// Simple Levenberg-Marquardt on a 2-parameter problem.
    fileprivate static func solveOptimalFocalShift(
        points: [Float], mask: [Bool]?,
        H: Int, W: Int, aspectRatio: Float
    ) -> (Float, Float) {
        let (uv, xyz) = gatherUVXYZ(
            points: points, mask: mask,
            H: H, W: W, aspectRatio: aspectRatio)
        if uv.count / 2 < 10 { return (1.0, 0.0) }

        var focal: Float = 1.0
        var shift: Float = 0.0
        var lambda: Float = 1e-3

        func cost(_ f: Float, _ s: Float) -> Float {
            var sum: Float = 0
            let n = uv.count / 2
            for i in 0..<n {
                let x = xyz[i * 3 + 0]
                let y = xyz[i * 3 + 1]
                let z = xyz[i * 3 + 2]
                let denom = z + s
                let ru = f * x / denom - uv[i * 2 + 0]
                let rv = f * y / denom - uv[i * 2 + 1]
                sum += ru * ru + rv * rv
            }
            return sum
        }

        var prevCost = cost(focal, shift)

        for _ in 0..<50 {
            // Build J^T J (2x2) and J^T r (2).
            var a00: Float = 0, a01: Float = 0, a11: Float = 0
            var b0: Float = 0, b1: Float = 0
            let n = uv.count / 2
            for i in 0..<n {
                let x = xyz[i * 3 + 0]
                let y = xyz[i * 3 + 1]
                let z = xyz[i * 3 + 2]
                let denom = z + shift
                let invD = 1 / denom
                // r_u = focal * x * invD - u
                let ru = focal * x * invD - uv[i * 2 + 0]
                let rv = focal * y * invD - uv[i * 2 + 1]

                // ∂r_u/∂focal = x * invD   ; ∂r_u/∂shift = -focal * x * invD²
                let dRuDf = x * invD
                let dRuDs = -focal * x * invD * invD
                let dRvDf = y * invD
                let dRvDs = -focal * y * invD * invD

                a00 += dRuDf * dRuDf + dRvDf * dRvDf
                a01 += dRuDf * dRuDs + dRvDf * dRvDs
                a11 += dRuDs * dRuDs + dRvDs * dRvDs
                b0 += dRuDf * ru + dRvDf * rv
                b1 += dRuDs * ru + dRvDs * rv
            }

            // Solve (JᵀJ + λ·diag(JᵀJ)) Δ = -Jᵀr
            let m00 = a00 * (1 + lambda)
            let m11 = a11 * (1 + lambda)
            let det = m00 * m11 - a01 * a01
            if abs(det) < 1e-20 { break }
            let df = (-b0 * m11 - -b1 * a01) / det
            let ds = (m00 * -b1 - a01 * -b0) / det

            let newFocal = focal + df
            let newShift = shift + ds
            let newCost = cost(newFocal, newShift)

            if newCost < prevCost {
                focal = newFocal
                shift = newShift
                lambda = max(lambda / 3, 1e-10)
                if abs(prevCost - newCost) < 1e-10 * max(prevCost, 1e-10) { break }
                prevCost = newCost
            } else {
                lambda = min(lambda * 5, 1e10)
            }
        }
        return (focal, shift)
    }

    /// Solve for `shift` alone given a known `focal`.
    fileprivate static func solveOptimalShift(
        points: [Float], mask: [Bool]?,
        H: Int, W: Int, aspectRatio: Float, focal: Float
    ) -> Float {
        let (uv, xyz) = gatherUVXYZ(
            points: points, mask: mask,
            H: H, W: W, aspectRatio: aspectRatio)
        if uv.count / 2 < 10 { return 0 }

        var shift: Float = 0
        var lambda: Float = 1e-3

        func cost(_ s: Float) -> Float {
            var sum: Float = 0
            let n = uv.count / 2
            for i in 0..<n {
                let x = xyz[i * 3 + 0]
                let y = xyz[i * 3 + 1]
                let z = xyz[i * 3 + 2]
                let denom = z + s
                let ru = focal * x / denom - uv[i * 2 + 0]
                let rv = focal * y / denom - uv[i * 2 + 1]
                sum += ru * ru + rv * rv
            }
            return sum
        }

        var prevCost = cost(shift)
        for _ in 0..<50 {
            var a: Float = 0
            var b: Float = 0
            let n = uv.count / 2
            for i in 0..<n {
                let x = xyz[i * 3 + 0]
                let y = xyz[i * 3 + 1]
                let z = xyz[i * 3 + 2]
                let denom = z + shift
                let invD = 1 / denom
                let ru = focal * x * invD - uv[i * 2 + 0]
                let rv = focal * y * invD - uv[i * 2 + 1]
                let dRuDs = -focal * x * invD * invD
                let dRvDs = -focal * y * invD * invD
                a += dRuDs * dRuDs + dRvDs * dRvDs
                b += dRuDs * ru + dRvDs * rv
            }
            let m = a * (1 + lambda)
            if m < 1e-20 { break }
            let ds = -b / m
            let newShift = shift + ds
            let newCost = cost(newShift)
            if newCost < prevCost {
                shift = newShift
                lambda = max(lambda / 3, 1e-10)
                if abs(prevCost - newCost) < 1e-10 * max(prevCost, 1e-10) { break }
                prevCost = newCost
            } else {
                lambda = min(lambda * 5, 1e10)
            }
        }
        return shift
    }
}
