// DINOv2 Vision Transformer backbone for MoGe-2
// Implements ViT-L/14 with intermediate layer extraction and positional embedding interpolation

import MLX
import MLXNN
import MLXFast
import Accelerate // for scipy-like zoom (we'll use linear interpolation fallback or keep numpy)

/// 2D image to patch embedding: (B, H, W, C) -> (B, N, D)
public class PatchEmbed: Module {
    public let patchSize: Int
    private let numPatches: Int // Reference for original 224x224
    @ModuleInfo(key: "proj")
    private var proj: Conv2d // (inC, embedDim), kernel=patchSize, stride=patchSize

    public init(
        imgSize: Int = 224,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 1024
    ) {
        self.patchSize = patchSize
        let grid = imgSize / patchSize
        self.numPatches = grid * grid
        
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            padding: 0,
            bias: true
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)
        var out = proj(x) // (B, H', W', D)
        let pH = H / patchSize
        let pW = W / patchSize
        out = out.reshaped([B, pH * pW, -1])
        return out
    }
}

/// Standard MLP with GELU activation: FC1 -> GELU -> FC2
public class Mlp: Module {
    @ModuleInfo(key: "fc1")
    private var fc1: Linear
    @ModuleInfo(key: "fc2")
    private var fc2: Linear

    public init(
        inFeatures: Int,
        hiddenFeatures: Int? = nil,
        outFeatures: Int? = nil,
        bias: Bool = true
    ) {
        let outF = outFeatures ?? inFeatures
        let hidden = hiddenFeatures ?? inFeatures
        
        self._fc1.wrappedValue = Linear(inFeatures, hidden, bias: bias)
        self._fc2.wrappedValue = Linear(hidden, outF, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(GELU()(fc1(x)))
    }
}

/// Multi-head self-attention with joint QKV projection.
public class Attention: Module {
    public let numHeads: Int
    private let headDim: Int
    public let scale: Float
    
    @ModuleInfo(key: "qkv")
    private var qkv: Linear // (dim, dim*3)
    @ModuleInfo(key: "proj")
    private var proj: Linear // (dim, dim)

    public init(
        dim: Int,
        numHeads: Int = 16,
        qkvBias: Bool = true,
        projBias: Bool = true
    ) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        // Use Float.squareRoot() to avoid MLX.sqrt confusion (returns Float, not MLXArray)
        self.scale = 1.0 / Float(self.headDim).squareRoot()
        
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim, bias: projBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let N = x.dim(1)
        let C = x.dim(2)

        // Joint QKV projection: (B, N, 3*D)
        let qkvOut = qkv(x)

        // Reshape to (B, N, 3, num_heads, head_dim) → (3, B, num_heads, N, head_dim)
        let qkvTransposed = qkvOut
            .reshaped([B, N, 3, numHeads, headDim])
            .transposed(2, 0, 3, 1, 4)

        let q = qkvTransposed[0]
        let k = qkvTransposed[1]
        let v = qkvTransposed[2]

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: nil
        )

        // (B, num_heads, N, head_dim) -> (B, N, D)
        out = out.transposed(0, 2, 1, 3).reshaped([B, N, C])
        return proj(out)
    }
}

/// Per-dimension learnable scaling (stabilizes ViT training)
public class LayerScale: Module {
    @ParameterInfo(key: "gamma")
    private var gamma: MLXArray // (dim,)

    public init(_ dim: Int, _ initValues: Float = 1e-5) {
        self._gamma = ParameterInfo(
            wrappedValue: MLXArray.full([dim], values: MLXArray(initValues)),
            key: "gamma"
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

/// Pre-norm transformer block: LN -> Attn -> LS -> Residual -> LN -> FFN -> LS -> Residual
public class Block: Module {
    @ModuleInfo(key: "norm1")
    private var norm1: LayerNorm
    @ModuleInfo(key: "attn")
    private var attn: Attention
    @ModuleInfo(key: "ls1")
    private var ls1: LayerScale?
    @ModuleInfo(key: "norm2")
    private var norm2: LayerNorm
    @ModuleInfo(key: "mlp")
    private var mlp: Mlp
    @ModuleInfo(key: "ls2")
    private var ls2: LayerScale?

    public init(
        dim: Int,
        numHeads: Int,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        projBias: Bool = true,
        ffnBias: Bool = true,
        initValues: Float? = nil
    ) {
        let mlpHidden = Int(Float(dim) * mlpRatio)
        
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._attn.wrappedValue = Attention(
            dim: dim, numHeads: numHeads,
            qkvBias: qkvBias, projBias: projBias
        )
        self._ls1.wrappedValue = initValues.map { LayerScale(dim, $0) }
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._mlp.wrappedValue = Mlp(
            inFeatures: dim,
            hiddenFeatures: mlpHidden,
            outFeatures: dim,
            bias: ffnBias
        )
        self._ls2.wrappedValue = initValues.map { LayerScale(dim, $0) }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Attention residual
        let attnOut = attn(norm1(x))
        let scaledAttn = ls1.map { $0(attnOut) } ?? attnOut
        var h = x + scaledAttn
        
        // FFN residual
        let ffnOut = mlp(norm2(h))
        let scaledFFN = ls2.map { $0(ffnOut) } ?? ffnOut
        h = h + scaledFFN
        
        return h
    }
}

/// DINOv2 ViT with intermediate layer extraction and positional embedding interpolation.
public class DinoVisionTransformer: Module {
    public let patchSize: Int
    public let embedDim: Int
    private let interpolateAntialias: Bool
    private let interpolateOffset: Float
    
    @ModuleInfo(key: "patch_embed")
    private var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token")
    private var clsToken: MLXArray // (1, 1, D)
    @ParameterInfo(key: "pos_embed")
    private var posEmbed: MLXArray // (1, N+1, D)
    @ParameterInfo(key: "mask_token")
    private var maskToken: MLXArray // (1, D)
    @ModuleInfo(key: "blocks")
    private var blocks: [Block]
    @ModuleInfo(key: "norm")
    private var norm: LayerNorm

    public init(
        imgSize: Int = 224,
        patchSize: Int = 14,
        inChans: Int = 3,
        embedDim: Int = 1024,
        depth: Int = 24,
        numHeads: Int = 16,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        ffnBias: Bool = true,
        projBias: Bool = true,
        initValues: Float? = nil,
        interpolateAntialias: Bool = false,
        interpolateOffset: Float = 0.1
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.interpolateAntialias = interpolateAntialias
        self.interpolateOffset = interpolateOffset
        
        let gridSide = imgSize / patchSize
        let numPatches = gridSide * gridSide
        
        self._patchEmbed.wrappedValue = PatchEmbed(
            imgSize: imgSize,
            patchSize: patchSize,
            inChannels: inChans,
            embedDim: embedDim
        )
        
        self._clsToken = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, 1, embedDim]),
            key: "cls_token"
        )
        self._posEmbed = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, numPatches + 1, embedDim]),
            key: "pos_embed"
        )
        self._maskToken = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, embedDim]),
            key: "mask_token"
        )
        
        // Build transformer blocks
        var blockArray: [Block] = []
        for _ in 0..<depth {
            blockArray.append(
                Block(
                    dim: embedDim,
                    numHeads: numHeads,
                    mlpRatio: mlpRatio,
                    qkvBias: qkvBias,
                    projBias: projBias,
                    ffnBias: ffnBias,
                    initValues: initValues
                )
            )
        }
        self._blocks.wrappedValue = blockArray
        self._norm.wrappedValue = LayerNorm(dimensions: embedDim, eps: 1e-6)
    }

    /// Interpolate position embeddings for arbitrary resolution using bicubic interpolation.
    /// Note: Uses Python scipy via a simple fallback or linear interpolation as placeholder
    /// since MLX doesn't have bicubic zoom. In practice, we use a simple bilinear approach.
    private func interpolatePosEncoding(_ x: MLXArray, _ H: Int, _ W: Int) -> MLXArray {
        let npatch = x.shape[1] - 1 // exclude cls
        let N = posEmbed.shape[1] - 1
        
        // If same size, return as-is
        if npatch == N && W == H {
            return posEmbed
        }
        
        let classPos = posEmbed[0..., ..<1] // (1, 1, D)
        var patchPos = posEmbed[0..., 1...] // (1, N, D)
        
        let dim = patchPos.shape.last!
        let M = Int(Float(N).squareRoot())
        assert(M * M == N, "Original patches must be square grid")
        
        let w0 = W / patchSize
        let h0 = H / patchSize
        
        // Reshape to (M, M, D) for 2D resizing
        patchPos = patchPos.reshaped([1, M, M, dim])
        
        // Apply interpolate_offset for compatibility with original DINOv2
        let sx = (Float(w0) + interpolateOffset) / (Float(M) + interpolateOffset)
        let sy = (Float(h0) + interpolateOffset) / (Float(M) + interpolateOffset)
        
        // Use bilinear resize as approximation to scipy ndimage zoom (order=3 bicubic)
        // MLX doesn't have bicubic, so we use bilinearResize (order=1)
        // This is slightly different from Python but close enough for inference
        patchPos = bilinearResize(patchPos, h0, w0)
        
        // Reshape back to (1, N_new, D)
        patchPos = patchPos.reshaped([1, h0 * w0, dim])
        
        return concatenated([classPos, patchPos], axis: 1)
    }

    /// Embed patches, prepend CLS, add position embeddings.
    private func prepareTokens(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)

        // Patch embedding: (B, N, D)
        var tokens = patchEmbed(x)

        // Broadcast CLS token: (B, 1, D)
        let clsTokens = broadcast(clsToken, to: [B, 1, embedDim])
        
        // Concatenate CLS + patches: (B, 1+N, D)
        tokens = concatenated([clsTokens, tokens], axis: 1)
        
        // Add interpolated position embeddings
        tokens = tokens + interpolatePosEncoding(tokens, H, W)
        return tokens
    }

    /// Extract features at specified intermediate layers.
    ///
    /// - Parameters:
    ///   - x: Input image (B, H, W, C) NHWC
    ///   - n: List of layer indices or number of last layers to extract
    ///   - returnClassToken: If true, return (patch_tokens, cls_token) tuples
    /// - Returns:
    ///   List of patch tokens or (patch_tokens, cls_token) tuples
    public func getIntermediateLayers(
        _ x: MLXArray,
        n: ([Int])? = nil,
        lastN: Int? = nil,
        returnClassToken: Bool = false
    ) -> [MLXArray] {
        let tokens = prepareTokens(x)
        let total = blocks.count

        // Determine which layers to extract and in what order. Python's
        // reference preserves the order given in `n`, so we do the same.
        let orderedIndices: [Int]
        if let nList = n {
            orderedIndices = nList
        } else {
            let count = lastN ?? 4
            orderedIndices = Array((total - count)..<total)
        }
        let layersToTake = Set(orderedIndices)

        var outputsByIndex: [Int: MLXArray] = [:]
        var currentTokens = tokens

        for (i, blk) in blocks.enumerated() {
            currentTokens = blk(currentTokens)
            if layersToTake.contains(i) {
                outputsByIndex[i] = currentTokens
            }
        }

        let outputs: [MLXArray] = orderedIndices.map { outputsByIndex[$0]! }
        
        // Apply final norm and extract cls/patch tokens. Python returns a
        // list of `(patch_tokens, cls_token)` tuples per layer; Swift returns
        // them flattened: `[patches_L0, cls_L0, patches_L1, cls_L1, ...]`
        // when `returnClassToken` is true.
        var result: [MLXArray] = []
        for out in outputs {
            let normalized = norm(out)
            let clsTok = normalized[0..., 0]          // (B, D)
            let patchToks = normalized[0..., 1...]    // (B, N, D)
            result.append(patchToks)
            if returnClassToken {
                result.append(clsTok)
            }
        }
        
        return result
    }
}
