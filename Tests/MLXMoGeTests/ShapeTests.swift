// Shape / sanity tests for MoGe-2 modules. These do not require a trained
// checkpoint — they exercise each module with random weights and verify the
// output tensor has the expected shape and dtype.

import XCTest
import MLX
import MLXNN
@testable import MLXMoGe

final class ShapeTests: XCTestCase {

    // MARK: - Image / geometry utilities

    func testBilinearResizeUpsample() {
        let x = MLXArray.zeros([1, 8, 8, 3])
        let y = bilinearResize(x, 32, 32)
        XCTAssertEqual(y.shape, [1, 32, 32, 3])
    }

    func testBilinearResizeDownsample() {
        let x = MLXArray.zeros([2, 64, 48, 4])
        let y = bilinearResize(x, 16, 12)
        XCTAssertEqual(y.shape, [2, 16, 12, 4])
    }

    func testBilinearResizeIdentity() {
        let x = MLXArray.ones([1, 10, 10, 3])
        let y = bilinearResize(x, 10, 10)
        XCTAssertEqual(y.shape, [1, 10, 10, 3])
    }

    func testPadReplicate() {
        let x = MLXArray.ones([1, 4, 4, 2])
        let y = padReplicate(x, 2)
        XCTAssertEqual(y.shape, [1, 8, 8, 2])
        // Padded values mirror edge (all ones here).
        XCTAssertEqual(y.sum().item(Float.self), Float(1 * 8 * 8 * 2))
    }

    func testNormalizedViewPlaneUV() {
        let uv = normalizedViewPlaneUV(width: 16, height: 12, aspectRatio: 4.0 / 3.0)
        XCTAssertEqual(uv.shape, [12, 16, 2])
        // Values must lie in [-spanX, spanX] × [-spanY, spanY].
        let aspectRatio: Float = 4.0 / 3.0
        let spanX = aspectRatio / (1 + aspectRatio * aspectRatio).squareRoot()
        let spanY = 1.0 / (1 + aspectRatio * aspectRatio).squareRoot()
        let minU = uv[.ellipsis, 0].min().item(Float.self)
        let maxU = uv[.ellipsis, 0].max().item(Float.self)
        let minV = uv[.ellipsis, 1].min().item(Float.self)
        let maxV = uv[.ellipsis, 1].max().item(Float.self)
        XCTAssertEqual(minU, -spanX, accuracy: 1e-5)
        XCTAssertEqual(maxU,  spanX, accuracy: 1e-5)
        XCTAssertEqual(minV, -spanY, accuracy: 1e-5)
        XCTAssertEqual(maxV,  spanY, accuracy: 1e-5)
    }

    func testIntrinsicsFromFocalCenter() {
        let K = intrinsicsFromFocalCenter(
            fx: [1.2, 0.8], fy: [1.1, 0.7],
            cx: [0.5, 0.5], cy: [0.5, 0.5])
        XCTAssertEqual(K.shape, [2, 3, 3])
        XCTAssertEqual(K[0, 0, 0].item(Float.self), 1.2, accuracy: 1e-6)
        XCTAssertEqual(K[1, 1, 1].item(Float.self), 0.7, accuracy: 1e-6)
        XCTAssertEqual(K[0, 2, 2].item(Float.self), 1.0, accuracy: 1e-6)
    }

    func testDepthMapToPointMap() {
        let depth = MLXArray.ones([1, 8, 8])
        let K = intrinsicsFromFocalCenter(fx: [1], fy: [1], cx: [0.5], cy: [0.5])
        let pts = depthMapToPointMap(depth: depth, intrinsics: K, height: 8, width: 8)
        XCTAssertEqual(pts.shape, [1, 8, 8, 3])
        // Z channel must equal the input depth.
        let z = pts[.ellipsis, 2]
        XCTAssertEqual(z.sum().item(Float.self), Float(8 * 8), accuracy: 1e-5)
    }

    // MARK: - DINOv2 modules

    func testPatchEmbed() {
        let pe = PatchEmbed(imgSize: 224, patchSize: 14, inChannels: 3, embedDim: 32)
        let x = MLXArray.zeros([1, 224, 224, 3])
        let y = pe(x)
        // 224/14 = 16 grid → 256 tokens
        XCTAssertEqual(y.shape, [1, 256, 32])
    }

    func testMlpForward() {
        let mlp = Mlp(inFeatures: 16, hiddenFeatures: 32, outFeatures: 16)
        let x = MLXArray.zeros([2, 10, 16])
        XCTAssertEqual(mlp(x).shape, [2, 10, 16])
    }

    func testAttentionForward() {
        let attn = Attention(dim: 32, numHeads: 4)
        let x = MLXArray.zeros([1, 20, 32])
        XCTAssertEqual(attn(x).shape, [1, 20, 32])
    }

    func testBlockForward() {
        let blk = Block(dim: 32, numHeads: 4, mlpRatio: 4.0, initValues: 1.0)
        let x = MLXArray.zeros([1, 20, 32])
        XCTAssertEqual(blk(x).shape, [1, 20, 32])
    }

    func testDinoVisionTransformerIntermediateLayers() {
        // Tiny ViT: depth=4, 2 heads, embedDim=16
        let vit = DinoVisionTransformer(
            imgSize: 112, patchSize: 14, inChans: 3,
            embedDim: 16, depth: 4, numHeads: 2, mlpRatio: 2.0,
            initValues: 1.0)
        let img = MLXArray.zeros([1, 112, 112, 3])  // 8x8 patch grid
        let feats = vit.getIntermediateLayers(img, returnClassToken: true)
        // 4 layers × 2 entries (patches + cls) = 8
        XCTAssertEqual(feats.count, 8)
        // patch tokens: (1, 64, 16)
        XCTAssertEqual(feats[0].shape, [1, 64, 16])
        // cls tokens: (1, 16)
        XCTAssertEqual(feats[1].shape, [1, 16])
    }

    // MARK: - Core modules (MoGeModules)

    func testResidualConvBlockShape() {
        let rcb = ResidualConvBlock(inChannels: 32, outChannels: 32, hiddenChannels: 32)
        let x = MLXArray.zeros([1, 16, 16, 32])
        XCTAssertEqual(rcb(x).shape, [1, 16, 16, 32])
    }

    func testResidualConvBlockChannelChange() {
        let rcb = ResidualConvBlock(inChannels: 16, outChannels: 32, hiddenChannels: 32)
        let x = MLXArray.zeros([1, 8, 8, 16])
        XCTAssertEqual(rcb(x).shape, [1, 8, 8, 32])
    }

    func testResamplerBilinear() {
        let rs = Resampler(inChannels: 16, outChannels: 8, type_: "bilinear", scaleFactor: 2)
        let x = MLXArray.zeros([1, 8, 8, 16])
        XCTAssertEqual(rs(x).shape, [1, 16, 16, 8])
    }

    func testResamplerConvTranspose() {
        let rs = Resampler(inChannels: 16, outChannels: 8, type_: "conv_transpose", scaleFactor: 2)
        let x = MLXArray.zeros([1, 8, 8, 16])
        XCTAssertEqual(rs(x).shape, [1, 16, 16, 8])
    }

    func testConvStackPyramid() {
        // 3-level pyramid, all bilinear resamplers, simple residual counts.
        // Channels must be ≥ 32 because the default hidden_norm is group_norm
        // which groups by `channels // 32` (matches the Python reference).
        let stack = ConvStack(
            dimIn: [64, nil, nil],
            dimResBlocks: [64, 64, 64],
            dimOut: [32, 32, 32],
            resamplers: ["bilinear", "bilinear"],
            numResBlocks: [1, 1, 1])
        let x0 = MLXArray.zeros([1, 8, 8, 64])
        // Levels 1 and 2 have dim_in=nil → inputs are not consumed (Identity
        // input block + hasInput=false). Pass dummies with matching H/W.
        let dummy1 = MLXArray.zeros([1, 16, 16, 1])
        let dummy2 = MLXArray.zeros([1, 32, 32, 1])
        let outs = stack([x0, dummy1, dummy2])
        XCTAssertEqual(outs.count, 3)
        XCTAssertEqual(outs[0].shape, [1, 8, 8, 32])
        XCTAssertEqual(outs[1].shape, [1, 16, 16, 32])
        XCTAssertEqual(outs[2].shape, [1, 32, 32, 32])
    }

    func testScaleHeadForward() {
        let sh = ScaleHead([1024, 256, 1])
        let x = MLXArray.zeros([2, 1024])
        XCTAssertEqual(sh(x).shape, [2, 1])
    }
}
