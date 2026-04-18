// DINOv2Encoder wrapper and MoGeModel main entry point
// Includes weight loading with key remapping for PyTorch checkpoints

import MLX
import MLXNN
import Foundation

/// DINOv2 backbone with intermediate feature extraction and 1x1 projections.
public class DINOv2Encoder: Module {
    public let patchSize: Int
    private let dimFeatures: Int
    /// Explicit layer indices to extract, in order. If `nil`, falls back to
    /// "last `intermediateLayerCount`" layers.
    private let intermediateLayerIndices: [Int]?
    private let intermediateLayerCount: Int

    @ModuleInfo(key: "backbone")
    private var backbone: DinoVisionTransformer
    @ModuleInfo(key: "output_projections")
    private var outputProjections: [Conv2d]

    // ImageNet normalization constants (NHWC)
    @ParameterInfo(key: "image_mean")
    private var imageMean: MLXArray // (1, 1, 1, 3)
    @ParameterInfo(key: "image_std")
    private var imageStd: MLXArray  // (1, 1, 1, 3)

    public init(
        backbone: String = "dinov2_vitl14",
        intermediateLayers: Any = 4, // Int (count, last N) or [Int] (explicit indices)
        dimOut: Int = 1024,
        embedDim: Int? = nil,
        depth: Int? = nil,
        numHeads: Int? = nil,
        patchSize: Int? = nil
    ) {
        if let idxList = intermediateLayers as? [Int] {
            self.intermediateLayerIndices = idxList
            self.intermediateLayerCount = idxList.count
        } else if let count = intermediateLayers as? Int {
            self.intermediateLayerIndices = nil
            self.intermediateLayerCount = count
        } else {
            fatalError("intermediateLayers must be Int or [Int], got \(type(of: intermediateLayers))")
        }
        // Backbone configs
        let configs: [String: [String: Any]] = [
            "dinov2_vits14": ["embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 14],
            "dinov2_vitb14": ["embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 14],
            "dinov2_vitl14": ["embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 14],
            "dinov2_vitg14": ["embed_dim": 1536, "depth": 40, "num_heads": 24, "patch_size": 14]
        ]
        
        guard let cfg = configs[backbone] else {
            fatalError("Unknown backbone: \(backbone)")
        }
        
        self.patchSize = Int(patchSize ?? cfg["patch_size"]! as! Int)
        self.dimFeatures = embedDim ?? Int(cfg["embed_dim"]! as! Int)
        
        let d = depth ?? Int(cfg["depth"]! as! Int)
        let nh = numHeads ?? Int(cfg["num_heads"]! as! Int)
        let ps = self.patchSize
        
        // Build backbone
        self._backbone.wrappedValue = DinoVisionTransformer(
            imgSize: 224,
            patchSize: ps,
            inChans: 3,
            embedDim: self.dimFeatures,
            depth: d,
            numHeads: nh,
            mlpRatio: 4.0,
            qkvBias: true,
            ffnBias: true,
            projBias: true,
            initValues: 1.0, // LayerScale init
            interpolateAntialias: false,
            interpolateOffset: 0.1
        )
        
        // Create output projections (1x1 convs), one per extracted layer
        var projs: [Conv2d] = []
        for _ in 0..<self.intermediateLayerCount {
            projs.append(
                Conv2d(inputChannels: self.dimFeatures, outputChannels: dimOut,
                      kernelSize: .init(1), bias: true)
            )
        }
        self._outputProjections.wrappedValue = projs
        
        // ImageNet normalization (NHWC order: RGB)
        self._imageMean = ParameterInfo(
            wrappedValue: MLXArray([0.485, 0.456, 0.406] as [Float]).reshaped([1, 1, 1, 3]),
            key: "image_mean"
        )
        self._imageStd = ParameterInfo(
            wrappedValue: MLXArray([0.229, 0.224, 0.225] as [Float]).reshaped([1, 1, 1, 3]),
            key: "image_std"
        )
    }

    public func callAsFunction(
        _ image: MLXArray,
        tokenRows: Int,
        tokenCols: Int,
        returnClassToken: Bool = false
    ) -> (MLXArray, MLXArray) {
        // Resize to target resolution
        let targetH = tokenRows * patchSize
        let targetW = tokenCols * patchSize
        var imageResized = bilinearResize(image, targetH, targetW)
        
        // Normalize with ImageNet stats
        imageResized = (imageResized - imageMean) / imageStd
        
        // Get intermediate layers from backbone. When the checkpoint specifies
        // explicit indices (e.g. `[5, 11, 17, 23]`) we must pass them through
        // — otherwise the default "last N" picks a different subset of layers
        // and the features no longer match the trained points head.
        // Returns [patches0, cls0, patches1, cls1, ...] when returnClassToken=true.
        let featuresList = backbone.getIntermediateLayers(
            imageResized,
            n: intermediateLayerIndices,
            lastN: intermediateLayerIndices == nil ? intermediateLayerCount : nil,
            returnClassToken: true
        )
        
        // Project through 1x1 convs and sum features from all levels
        var projectedSum: MLXArray?
        var lastClsToken: MLXArray? = nil
        
        // featuresList structure: [patches0, cls0, patches1, cls1, ...]
        for i in stride(from: 0, to: featuresList.count, by: 2) {
            let feat = featuresList[i] // (B, N, D)
            lastClsToken = (featuresList.count > i + 1) ? featuresList[i+1] : nil
            
            let B = feat.dim(0)
            // Reshape to (B, token_rows, token_cols, D) then conv
            let feat2D = feat.reshaped([B, tokenRows, tokenCols, -1])
            let projected = outputProjections[i / 2](feat2D)
            
            if i == 0 {
                projectedSum = projected
            } else {
                projectedSum = projectedSum! + projected
            }
        }
        
        let x = projectedSum!
        // Return (features, cls_token) - last layer's CLS token
        return (x, lastClsToken!)
    }
}

/// MoGe-2: Monocular Geometry Estimation model.
/// Estimates depth, surface normals, and camera intrinsics from a single image.
public class MoGeModel: Module {
    @ModuleInfo(key: "encoder")
    private var encoder: DINOv2Encoder
    @ModuleInfo(key: "neck")
    private var neck: ConvStack
    
    @ModuleInfo(key: "points_head")
    private var pointsHead: ConvStack?
    @ModuleInfo(key: "normal_head")
    private var normalHead: ConvStack?
    @ModuleInfo(key: "mask_head")
    private var maskHead: ConvStack?
    @ModuleInfo(key: "scale_head")
    private var scaleHead: ScaleHead?
    
    public let remapOutput: String
    public let numTokensRange: [Int] // [min, max]

    public init(
        encoder: [String: Any],
        neck: [String: Any],
        pointsHead: [String: Any]? = nil,
        normalHead: [String: Any]? = nil,
        maskHead: [String: Any]? = nil,
        scaleHead: [String: Any]? = nil,
        remapOutput: String = "linear",
        numTokensRange: [Int] = [1200, 3600]
    ) {
        self.remapOutput = remapOutput
        self.numTokensRange = numTokensRange
        
        // Parse encoder config. `intermediate_layers` may be an Int (count of
        // last-N layers) or a list of explicit indices (e.g. [5,11,17,23]).
        let encBackbone = encoder["backbone"] as? String ?? "dinov2_vitl14"
        let encIntermediate: Any
        if let list = encoder["intermediate_layers"] as? [Int] {
            encIntermediate = list
        } else if let n = encoder["intermediate_layers"] as? Int {
            encIntermediate = n
        } else {
            encIntermediate = 4
        }
        let encDimOut = encoder["dim_out"] as? Int ?? 1024

        self._encoder.wrappedValue = DINOv2Encoder(
            backbone: encBackbone,
            intermediateLayers: encIntermediate,
            dimOut: encDimOut
        )
        
        // Parse neck config
        let nDimIn = neck["dim_in"] as? [Int?] ?? []
        let nDimRes = neck["dim_res_blocks"] as? [Int] ?? []
        let nDimOut = neck["dim_out"] as? [Int?] ?? nil
        let nResamplers = neck["resamplers"] as? [String] ?? nil
        let nDimTimesHidden = neck["dim_times_res_block_hidden"] as? Int ?? 1
        let nNumRes = neck["num_res_blocks"]
        let nActivation = neck["activation"] as? String ?? "relu"
        let nResInNorm = neck["res_block_in_norm"] as? String ?? "layer_norm"
        let nResHiddenNorm = neck["res_block_hidden_norm"] as? String ?? "group_norm"
        
        var numResBlocks: [Int]? = nil
        if let nr = nNumRes as? Int {
            numResBlocks = Array(repeating: nr, count: nDimRes.count)
        } else if let nr = nNumRes as? [Int] {
            numResBlocks = nr
        }
        
        self._neck.wrappedValue = ConvStack(
            dimIn: nDimIn,
            dimResBlocks: nDimRes,
            dimOut: nDimOut,
            resamplers: nResamplers,
            dimTimesResBlockHidden: nDimTimesHidden,
            numResBlocks: numResBlocks ?? [],
            resBlockInNorm: nResInNorm,
            resBlockHiddenNorm: nResHiddenNorm,
            activation: nActivation
        )
        
        // Optional heads
        self._pointsHead = ModuleInfo(
            wrappedValue: pointsHead.map { Self.makeConvStack($0) },
            key: "points_head"
        )
        self._normalHead = ModuleInfo(
            wrappedValue: normalHead.map { Self.makeConvStack($0) },
            key: "normal_head"
        )
        self._maskHead = ModuleInfo(
            wrappedValue: maskHead.map { Self.makeConvStack($0) },
            key: "mask_head"
        )
        let scaleHeadModule: ScaleHead? = {
            guard let sh = scaleHead, let dims = sh["dims"] as? [Int] else { return nil }
            return ScaleHead(dims)
        }()
        self._scaleHead = ModuleInfo(wrappedValue: scaleHeadModule, key: "scale_head")
    }

    private static func makeConvStack(_ config: [String: Any]) -> ConvStack {
        let dimIn = config["dim_in"] as? [Int?] ?? []
        let dimRes = config["dim_res_blocks"] as? [Int] ?? []
        let dimOut = config["dim_out"] as? [Int?] ?? nil
        let resamplers = config["resamplers"] as? [String] ?? nil
        let dimTimesHidden = config["dim_times_res_block_hidden"] as? Int ?? 1
        let activation = config["activation"] as? String ?? "relu"
        let resInNorm = config["res_block_in_norm"] as? String ?? "layer_norm"
        let resHiddenNorm = config["res_block_hidden_norm"] as? String ?? "group_norm"

        var numResBlocks: [Int]? = nil
        if let nr = config["num_res_blocks"] as? Int {
            numResBlocks = Array(repeating: nr, count: dimRes.count)
        } else if let nr = config["num_res_blocks"] as? [Int] {
            numResBlocks = nr
        }

        return ConvStack(
            dimIn: dimIn,
            dimResBlocks: dimRes,
            dimOut: dimOut,
            resamplers: resamplers,
            dimTimesResBlockHidden: dimTimesHidden,
            numResBlocks: numResBlocks ?? [],
            resBlockInNorm: resInNorm,
            resBlockHiddenNorm: resHiddenNorm,
            activation: activation
        )
    }

    /// Output remapping (linear, exp, sinh, etc.)
    private func remapPoints(_ points: MLXArray) -> MLXArray {
        switch remapOutput {
            case "linear":
                return points
            case "exp":
                let xy = points[.ellipsis, ..<2]
                var z = points[.ellipsis, 2...]
                z = MLX.exp(z)
                return concatenated([xy * z, z], axis: -1)
            case "sinh":
                return MLX.sinh(points)
            case "sinh_exp":
                let xy = MLX.sinh(points[.ellipsis, ..<2])
                var z = points[.ellipsis, 2...]
                z = MLX.exp(z)
                return concatenated([xy, z], axis: -1)
            default:
                fatalError("Unknown remap_output: \(remapOutput)")
        }
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - image: (B, H, W, 3) NHWC float [0, 1]
    ///   - numTokens: Target number of ViT tokens
    /// - Returns:
    ///   Dict with 'points', 'normal', 'mask', 'metric_scale' (present if head exists)
    public func callAsFunction(_ image: MLXArray, _ numTokens: Int) -> [String: MLXArray] {
        let B = image.dim(0)
        let imgH = image.dim(1)
        let imgW = image.dim(2)
        let aspectRatio = Float(imgW) / Float(imgH)
        
        // Compute token grid dimensions
        let baseH = Int(sqrt(Float(numTokens) / aspectRatio))
        let baseW = Int(sqrt(Float(numTokens) * aspectRatio))
        
        // Encode: returns (features, cls_token)
        let (features, clsToken) = encoder(image, tokenRows: baseH, tokenCols: baseW)
        
        // Build 5-level feature pyramid with UV coordinates
        var pyramid: [MLXArray] = []
        
        for level in 0..<5 {
            let h = (level == 0) ? baseH : baseH * (1 << level)
            let w = (level == 0) ? baseW : baseW * (1 << level)

            // (H, W, 2) UV grid → broadcast to (B, H, W, 2)
            var uv = normalizedViewPlaneUV(width: w, height: h, aspectRatio: aspectRatio)
            uv = broadcast(uv.expandedDimensions(axis: 0), to: [B, h, w, 2])

            if level == 0 {
                pyramid.append(concatenated([features, uv], axis: -1))
            } else {
                pyramid.append(uv)
            }
        }
        
        // Process through neck
        let neckFeatures = neck(pyramid)
        
        var output: [String: MLXArray] = [:]
        
        // Points head (3D coordinates)
        if let ph = pointsHead {
            var points = ph(neckFeatures).last!
            points = bilinearResize(points, imgH, imgW)
            points = remapPoints(points)
            output["points"] = points
        }
        
        // Normal head (surface normals)
        if let nh = normalHead {
            var normal = nh(neckFeatures).last!
            normal = bilinearResize(normal, imgH, imgW)
            // L2 normalize along channel axis
            let norm2 = MLX.sqrt((normal * normal).sum(axis: -1, keepDims: true)) + 1e-8
            normal = normal / norm2
            output["normal"] = normal
        }

        // Mask head (validity mask)
        if let mh = maskHead {
            var mask = mh(neckFeatures).last!
            mask = bilinearResize(mask, imgH, imgW)
            // Sigmoid on first channel → (B, H, W)
            mask = MLX.sigmoid(mask[.ellipsis, 0])
            output["mask"] = mask
        }
        
        // Scale head (metric scale from CLS token)
        if let sh = scaleHead {
            var scale = sh(clsToken)
            scale = MLX.exp(scale).squeezed(axis: -1)
            output["metric_scale"] = scale
        }
        
        return output
    }

    /// Load a model from a converted weights directory (config.json +
    /// weights.safetensors). See ``WeightLoading`` for key-remap details.
    public static func fromPretrained(
        path: String,
        dtype: DType = .float16
    ) throws -> MoGeModel {
        return try fromPretrained(url: URL(fileURLWithPath: path), dtype: dtype)
    }

    public static func fromPretrained(
        url: URL,
        dtype: DType = .float16
    ) throws -> MoGeModel {
        return try MoGeWeightLoader.load(path: url.path, dtype: dtype)
    }
}
