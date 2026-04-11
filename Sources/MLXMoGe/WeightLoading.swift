// Weight loading and key remapping for MoGe-2 checkpoints converted with
// `mlx_moge/convert.py`. Loads config.json + weights.safetensors, remaps
// PyTorch `nn.Sequential` keys to the Swift module tree, casts to the target
// dtype, and assigns onto a freshly-constructed `MoGeModel`.

import Foundation
import MLX
import MLXNN

public enum MoGeWeightLoadingError: Error, CustomStringConvertible {
    case missingFile(String)
    case invalidConfig

    public var description: String {
        switch self {
        case .missingFile(let p): return "Missing file: \(p)"
        case .invalidConfig:      return "Failed to parse config.json"
        }
    }
}

public enum MoGeWeightLoader {

    /// Load a `MoGeModel` from a directory containing `config.json` and
    /// `weights.safetensors`.
    public static func load(path: String, dtype: DType = .float16) throws -> MoGeModel {
        let dir = URL(fileURLWithPath: path)
        let configURL  = dir.appendingPathComponent("config.json")
        let weightsURL = dir.appendingPathComponent("weights.safetensors")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MoGeWeightLoadingError.missingFile(configURL.path)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw MoGeWeightLoadingError.missingFile(weightsURL.path)
        }

        let configData = try Data(contentsOf: configURL)
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw MoGeWeightLoadingError.invalidConfig
        }

        let model = MoGeModel(
            encoder: config["encoder"] as? [String: Any] ?? [:],
            neck:    config["neck"] as? [String: Any] ?? [:],
            pointsHead: config["points_head"] as? [String: Any],
            normalHead: config["normal_head"] as? [String: Any],
            maskHead:   config["mask_head"]   as? [String: Any],
            scaleHead:  config["scale_head"]  as? [String: Any],
            remapOutput: config["remap_output"] as? String ?? "linear",
            numTokensRange: config["num_tokens_range"] as? [Int] ?? [1200, 3600]
        )

        let weights = try MLX.loadArrays(url: weightsURL)

        var remapped: [(String, MLXArray)] = []
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            let newKey = remapWeightKey(key)
            remapped.append((newKey, value.asType(dtype)))
        }

        MLX.eval(remapped.map { $0.1 })
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: [.noUnusedKeys])
        return model
    }

    /// Remap PyTorch checkpoint key to MLX Swift module path.
    ///
    /// Structural differences handled here (mirrors `_remap_weight_key` in
    /// `mlx_moge/model/moge.py`):
    /// - `*.resamplers.{i}.{j}.param` → `*.resamplers.{i}.layers.{j}.param`
    /// - `scale_head.{j}.param`       → `scale_head.layers.{j}.param`
    static func remapWeightKey(_ key: String) -> String {
        // Resampler: insert `.layers` between resampler index and child index
        if let range = key.range(of: #"\.resamplers\.\d+\."#, options: .regularExpression) {
            let prefix = String(key[..<range.upperBound])   // up to and incl. "resamplers.N."
            let rest   = String(key[range.upperBound...])   // "J.param..."
            if let dot = rest.firstIndex(of: ".") {
                let childIdx = rest[..<dot]
                let param    = rest[rest.index(after: dot)...]
                return "\(prefix)layers.\(childIdx).\(param)"
            }
        }

        // ScaleHead: `scale_head.{j}.param` → `scale_head.layers.{j}.param`
        if key.hasPrefix("scale_head.") {
            let tail = key.dropFirst("scale_head.".count)
            if let dot = tail.firstIndex(of: "."),
               Int(tail[..<dot]) != nil {
                return "scale_head.layers.\(tail)"
            }
        }

        return key
    }
}
