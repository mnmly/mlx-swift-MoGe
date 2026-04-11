// Core modules for MoGe-2: ResidualConvBlock, ConvTranspose2dModule, Resampler,
// ConvStack (feature pyramid), and ScaleHead.
//
// All operations are NHWC (MLX native). Weight key layout follows the Python
// reference in mlx_moge/model/modules.py, with one structural remap handled at
// load time for `layers.<i>` nested inside Sequential-style lists.

import MLX
import MLXNN

// MARK: - Factories

fileprivate func makeNorm(_ normType: String, _ channels: Int) -> UnaryLayer {
    switch normType {
    case "none":
        return Identity()
    case "group_norm":
        return GroupNorm(groupCount: channels / 32, dimensions: channels)
    case "layer_norm":
        // GroupNorm with 1 group is equivalent to LayerNorm over channels.
        return GroupNorm(groupCount: 1, dimensions: channels)
    default:
        fatalError("Unsupported norm type: \(normType)")
    }
}

fileprivate func makeActivation(_ actType: String) -> UnaryLayer {
    switch actType {
    case "relu":
        return ReLU()
    case "leaky_relu":
        return LeakyReLU()
    case "silu":
        return SiLU()
    case "elu":
        return ELU()
    default:
        fatalError("Unsupported activation: \(actType)")
    }
}

// MARK: - ResidualConvBlock

/// Residual block: [Norm -> Act -> Conv] x2 + skip.
///
/// Weight keys (matching PyTorch nn.Sequential):
///   layers.0 norm1, layers.1 act, layers.2 conv1,
///   layers.3 norm2, layers.4 act, layers.5 conv2.
public class ResidualConvBlock: Module, UnaryLayer {
    public let pad: Int

    @ModuleInfo(key: "layers") public var layers: [UnaryLayer]
    @ModuleInfo(key: "skip") public var skip: Conv2d?

    public init(
        inChannels: Int,
        outChannels: Int? = nil,
        hiddenChannels: Int? = nil,
        kernelSize: Int = 3,
        activation: String = "relu",
        inNorm: String = "layer_norm",
        hiddenNorm: String = "group_norm"
    ) {
        let outCh = outChannels ?? inChannels
        let hiddenCh = hiddenChannels ?? inChannels
        self.pad = kernelSize / 2

        let built: [UnaryLayer] = [
            makeNorm(inNorm, inChannels),
            makeActivation(activation),
            Conv2d(inputChannels: inChannels, outputChannels: hiddenCh,
                   kernelSize: .init(kernelSize), padding: .init(0), bias: true),
            makeNorm(hiddenNorm, hiddenCh),
            makeActivation(activation),
            Conv2d(inputChannels: hiddenCh, outputChannels: outCh,
                   kernelSize: .init(kernelSize), padding: .init(0), bias: true),
        ]
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")

        let skipConv: Conv2d? = (inChannels != outCh)
            ? Conv2d(inputChannels: inChannels, outputChannels: outCh,
                     kernelSize: .init(1), bias: true)
            : nil
        self._skip = ModuleInfo(wrappedValue: skipConv, key: "skip")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual: MLXArray = (skip != nil) ? skip!(x) : x

        var h = layers[0](x)                     // norm1
        h = layers[1](h)                          // act
        h = padReplicate(h, pad)
        h = layers[2](h)                          // conv1

        h = layers[3](h)                          // norm2
        h = layers[4](h)                          // act
        h = padReplicate(h, pad)
        h = layers[5](h)                          // conv2

        return h + residual
    }
}

// MARK: - ConvTranspose2dModule

/// Learned transposed convolution for 2x upsampling. Weights are pre-flipped
/// spatially during conversion and stored `(outC, kH, kW, inC)` so that
/// `convGeneral` with `inputDilation = stride` produces the correct output.
public class ConvTranspose2dModule: Module, UnaryLayer {
    public let stride: Int

    public let weight: MLXArray
    public let bias: MLXArray

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 2,
        stride: Int = 2
    ) {
        self.stride = stride
        self.weight = MLXArray.zeros([outChannels, kernelSize, kernelSize, inChannels])
        self.bias = MLXArray.zeros([outChannels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        convTranspose2d(x, weight: weight, bias: bias, stride: stride)
    }
}

// MARK: - Resampler

/// Spatial resampler between feature pyramid levels.
///
/// Weight keys:
///   layers.0 ConvTranspose2dModule (conv_transpose) or Identity (bilinear)
///   layers.1 Conv2d post-upsample 3x3 (replicate padded)
public class Resampler: Module, UnaryLayer {
    public let type_: String
    public let scaleFactor: Int

    @ModuleInfo(key: "layers") public var layers: [UnaryLayer]

    public init(
        inChannels: Int,
        outChannels: Int,
        type_: String,
        scaleFactor: Int = 2
    ) {
        self.type_ = type_
        self.scaleFactor = scaleFactor

        let built: [UnaryLayer]
        switch type_ {
        case "conv_transpose":
            built = [
                ConvTranspose2dModule(
                    inChannels: inChannels,
                    outChannels: outChannels,
                    kernelSize: scaleFactor,
                    stride: scaleFactor
                ),
                Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                       kernelSize: .init(3), padding: .init(0), bias: true),
            ]
        case "bilinear":
            built = [
                Identity(),
                Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                       kernelSize: .init(3), padding: .init(0), bias: true),
            ]
        default:
            fatalError("Unsupported resampler type: \(type_)")
        }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        switch type_ {
        case "conv_transpose":
            y = layers[0](x)
        case "bilinear":
            let H = x.dim(1)
            let W = x.dim(2)
            y = bilinearResize(x, H * scaleFactor, W * scaleFactor)
        default:
            fatalError("Unsupported resampler type: \(type_)")
        }
        y = padReplicate(y, 1)
        return layers[1](y)
    }
}

// MARK: - ConvStack

/// Multi-scale feature pyramid processor with progressive upsampling.
public class ConvStack: Module {
    public let numLevels: Int

    @ModuleInfo(key: "input_blocks") public var inputBlocks: [UnaryLayer]
    @ModuleInfo(key: "res_blocks") public var resBlocks: [[ResidualConvBlock]]
    @ModuleInfo(key: "resamplers") public var resamplers: [Resampler]
    @ModuleInfo(key: "output_blocks") public var outputBlocks: [UnaryLayer]

    /// Whether the corresponding input block is a real 1x1 Conv (`true`) or an
    /// Identity/none placeholder (`false`). Used to decide whether to add the
    /// per-level skip connection at forward time.
    private let hasInput: [Bool]

    public init(
        dimIn: [Int?],
        dimResBlocks: [Int],
        dimOut: [Int?]? = nil,
        resamplers: [String]? = nil,
        dimTimesResBlockHidden: Int = 1,
        numResBlocks: [Int]? = nil,
        resBlockInNorm: String = "layer_norm",
        resBlockHiddenNorm: String = "group_norm",
        activation: String = "relu"
    ) {
        let numLevels = dimResBlocks.count
        self.numLevels = numLevels

        let dimOutArr: [Int?] = dimOut ?? Array(repeating: nil, count: numLevels)
        let numResArr: [Int] = numResBlocks
            ?? Array(repeating: 1, count: numLevels)
        let resamplerTypes: [String] = resamplers
            ?? Array(repeating: "bilinear", count: max(0, numLevels - 1))

        // Input blocks
        var inputBuilt: [UnaryLayer] = []
        var hasInputBuilt: [Bool] = []
        for i in 0..<numLevels {
            if let dIn = dimIn[i] {
                inputBuilt.append(
                    Conv2d(inputChannels: dIn, outputChannels: dimResBlocks[i],
                           kernelSize: .init(1), bias: true)
                )
                hasInputBuilt.append(true)
            } else {
                inputBuilt.append(Identity())
                hasInputBuilt.append(false)
            }
        }
        self._inputBlocks = ModuleInfo(wrappedValue: inputBuilt, key: "input_blocks")
        self.hasInput = hasInputBuilt

        // Residual blocks per level (empty inner array => no blocks, no keys)
        var resBuilt: [[ResidualConvBlock]] = []
        for i in 0..<numLevels {
            let count = numResArr[i]
            var level: [ResidualConvBlock] = []
            for _ in 0..<count {
                level.append(
                    ResidualConvBlock(
                        inChannels: dimResBlocks[i],
                        hiddenChannels: dimResBlocks[i] * dimTimesResBlockHidden,
                        kernelSize: 3,
                        activation: activation,
                        inNorm: resBlockInNorm,
                        hiddenNorm: resBlockHiddenNorm
                    )
                )
            }
            resBuilt.append(level)
        }
        self._resBlocks = ModuleInfo(wrappedValue: resBuilt, key: "res_blocks")

        // Resamplers between levels
        var resamplersBuilt: [Resampler] = []
        for i in 0..<(numLevels - 1) {
            resamplersBuilt.append(
                Resampler(
                    inChannels: dimResBlocks[i],
                    outChannels: dimResBlocks[i + 1],
                    type_: resamplerTypes[i]
                )
            )
        }
        self._resamplers = ModuleInfo(wrappedValue: resamplersBuilt, key: "resamplers")

        // Output blocks
        var outBuilt: [UnaryLayer] = []
        for i in 0..<numLevels {
            if let dOut = dimOutArr[i] {
                outBuilt.append(
                    Conv2d(inputChannels: dimResBlocks[i], outputChannels: dOut,
                           kernelSize: .init(1), bias: true)
                )
            } else {
                outBuilt.append(Identity())
            }
        }
        self._outputBlocks = ModuleInfo(wrappedValue: outBuilt, key: "output_blocks")
    }

    public func callAsFunction(_ inFeatures: [MLXArray]) -> [MLXArray] {
        var outFeatures: [MLXArray] = []
        var x: MLXArray! = nil

        for i in 0..<numLevels {
            let feature = inputBlocks[i](inFeatures[i])

            if i == 0 {
                x = feature
            } else if hasInput[i] {
                x = x + feature
            }

            for blk in resBlocks[i] {
                x = blk(x)
            }

            outFeatures.append(outputBlocks[i](x))

            if i < numLevels - 1 {
                x = resamplers[i](x)
            }
        }

        return outFeatures
    }
}

// MARK: - ScaleHead

/// MLP for metric scale prediction: Linear -> ReLU -> ... -> Linear.
///
/// Weight keys: layers.0 Linear, layers.1 ReLU, layers.2 Linear, ...
public class ScaleHead: Module, UnaryLayer {
    @ModuleInfo(key: "layers") public var layers: [UnaryLayer]

    public init(_ dims: [Int]) {
        var built: [UnaryLayer] = []
        for i in 0..<(dims.count - 1) {
            built.append(Linear(dims[i], dims[i + 1], bias: true))
            if i < dims.count - 2 {
                built.append(ReLU())
            }
        }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for l in layers {
            h = l(h)
        }
        return h
    }
}
