// Geometry utilities for MoGe-2 post-processing.
//
// Pure-MLX helpers (UV grids, intrinsics, unprojection) live here. The
// focal/shift recovery that Python implements with `scipy.optimize.least_squares`
// is ported to Swift in MoGeInference.swift as a small Levenberg-Marquardt solver.

import MLX
import Foundation

/// Generate normalized UV coordinates for a view plane, matching
/// `normalized_view_plane_uv` in `mlx_moge/utils/geometry.py`.
///
/// - Parameters:
///   - width: Grid width
///   - height: Grid height
///   - aspectRatio: Image width / height
///   - dtype: Output dtype
/// - Returns: `(height, width, 2)` tensor with UV in
///   `[-spanX, +spanX] x [-spanY, +spanY]`.
public func normalizedViewPlaneUV(
    width: Int,
    height: Int,
    aspectRatio: Float,
    dtype: DType = .float32
) -> MLXArray {
    let spanX = aspectRatio / (1.0 + aspectRatio * aspectRatio).squareRoot()
    let spanY = 1.0 / (1.0 + aspectRatio * aspectRatio).squareRoot()

    let u = MLX.linspace(-spanX, spanX, count: width).asType(dtype) // (W,)
    let v = MLX.linspace(-spanY, spanY, count: height).asType(dtype) // (H,)

    // meshgrid(v, u, indexing="ij") -> (gridV, gridU) both (H, W)
    let gridU = broadcast(u.expandedDimensions(axis: 0), to: [height, width]) // (H, W)
    let gridV = broadcast(v.expandedDimensions(axis: 1), to: [height, width]) // (H, W)

    return stacked([gridU, gridV], axis: -1) // (H, W, 2)
}

/// Build a `(B, 3, 3)` intrinsics matrix from per-sample focal/principal arrays.
public func intrinsicsFromFocalCenter(
    fx: [Float],
    fy: [Float],
    cx: [Float],
    cy: [Float]
) -> MLXArray {
    let B = fx.count
    precondition(fy.count == B && cx.count == B && cy.count == B)
    var rows: [[Float]] = []
    for i in 0..<B {
        rows.append([fx[i], 0, cx[i],
                     0, fy[i], cy[i],
                     0,     0,    1])
    }
    let flat = rows.flatMap { $0 }
    return MLXArray(flat).reshaped([B, 3, 3])
}

/// Unproject a depth map to a 3D point cloud using camera intrinsics.
///
/// - Parameters:
///   - depth: `(B, H, W)` depth values
///   - intrinsics: `(B, 3, 3)` normalized intrinsics (cx/cy in [0,1])
///   - height: Image height
///   - width: Image width
/// - Returns: `(B, H, W, 3)` point cloud
public func depthMapToPointMap(
    depth: MLXArray,
    intrinsics: MLXArray,
    height: Int,
    width: Int
) -> MLXArray {
    let isBatched = depth.ndim == 3
    let depthBatched = isBatched ? depth : depth.expandedDimensions(axis: 0)
    let intrinsicsBatched = intrinsics.ndim == 3 ? intrinsics : intrinsics.expandedDimensions(axis: 0)

    let B = depthBatched.dim(0)
    let H = height
    let W = width

    // Pixel coordinates in [0, 1] with half-pixel offsets.
    let u = MLX.linspace(0.5 / Float(W), 1 - 0.5 / Float(W), count: W) // (W,)
    let v = MLX.linspace(0.5 / Float(H), 1 - 0.5 / Float(H), count: H) // (H,)

    let gridU = broadcast(u.expandedDimensions(axis: 0), to: [H, W]) // (H, W)
    let gridV = broadcast(v.expandedDimensions(axis: 1), to: [H, W]) // (H, W)

    let gridU4 = gridU.reshaped([1, H, W, 1])
    let gridV4 = gridV.reshaped([1, H, W, 1])

    let fx = intrinsicsBatched[0..., 0, 0].reshaped([B, 1, 1, 1])
    let fy = intrinsicsBatched[0..., 1, 1].reshaped([B, 1, 1, 1])
    let cx = intrinsicsBatched[0..., 0, 2].reshaped([B, 1, 1, 1])
    let cy = intrinsicsBatched[0..., 1, 2].reshaped([B, 1, 1, 1])

    let depth4 = depthBatched.expandedDimensions(axis: -1) // (B, H, W, 1)

    let x = (gridU4 - cx) / fx * depth4
    let y = (gridV4 - cy) / fy * depth4
    let z = depth4

    var result = concatenated([x, y, z], axis: -1) // (B, H, W, 3)
    if !isBatched {
        result = result[0]
    }
    return result
}

// MARK: - PLY Export

/// Represents a vertex in a PLY file with optional color.
public struct PLYVertex {
    public var x: Float
    public var y: Float
    public var z: Float
    public var nx: Float?
    public var ny: Float?
    public var nz: Float?
    public var r: UInt8?
    public var g: UInt8?
    public var b: UInt8?

    public init(
        x: Float, y: Float, z: Float,
        nx: Float? = nil, ny: Float? = nil, nz: Float? = nil,
        r: UInt8? = nil, g: UInt8? = nil, b: UInt8? = nil
    ) {
        self.x = x; self.y = y; self.z = z
        self.nx = nx; self.ny = ny; self.nz = nz
        self.r = r; self.g = g; self.b = b
    }
}

/// Represents a face (triangle) in a PLY file.
public struct PLYFace {
    public var indices: [Int]

    public init(_ i0: Int, _ i1: Int, _ i2: Int) {
        self.indices = [i0, i1, i2]
    }
}

/// Export a point cloud to PLY format.
///
/// - Parameters:
///   - path: Output file path
///   - vertices: Array of vertices
///   - faces: Optional array of triangular faces
///   - binary: Whether to write in binary format (true) or ASCII (false)
/// - Throws: Error on write failure
public func exportPLY(
    to path: String,
    vertices: [PLYVertex],
    faces: [PLYFace]? = nil,
    binary: Bool = true
) throws {
    let header = generatePLYHeader(vertices: vertices, faces: faces, binary: binary)
    var data = Data()
    if binary {
        // Binary PLY uses little-endian
        for v in vertices {
            let vx = withUnsafeBytes(of: v.x) { Data($0) }
            let vy = withUnsafeBytes(of: v.y) { Data($0) }
            let vz = withUnsafeBytes(of: v.z) { Data($0) }
            data.append(vx)
            data.append(vy)
            data.append(vz)
            
            if let nx = v.nx {
                let nxf = withUnsafeBytes(of: nx) { Data($0) }
                data.append(nxf)
            }
            if let ny = v.ny {
                let nyf = withUnsafeBytes(of: ny) { Data($0) }
                data.append(nyf)
            }
            if let nz = v.nz {
                let nzf = withUnsafeBytes(of: nz) { Data($0) }
                data.append(nzf)
            }
            
            if let r = v.r { data.append(r) }
            if let g = v.g { data.append(g) }
            if let b = v.b { data.append(b) }
        }
        
        if let faces = faces {
            for face in faces {
                precondition(face.indices.count == 3, "PLY only supports triangles")
                let count: UInt8 = 3
                data.append(count)
                for idx in face.indices {
                    let leIdx = idx.toLittleEndian()
                    let idxData = withUnsafeBytes(of: leIdx) { Data($0) }
                    data.append(idxData)
                }
            }
        }
    } else {
        // ASCII PLY
        var writer = ""
        for v in vertices {
            writer += "\(v.x) \(v.y) \(v.z)"
            if let nx = v.nx { writer += " \(nx)" }
            if let ny = v.ny { writer += " \(ny)" }
            if let nz = v.nz { writer += " \(nz)" }
            if let r = v.r, let g = v.g, let b = v.b {
                writer += " \(r) \(g) \(b)"
            }
            writer += "\n"
        }
        
        if let faces = faces {
            for face in faces {
                writer.append("3 \(face.indices[0]) \(face.indices[1]) \(face.indices[2])\n")
            }
        }

        data = writer.data(using: .utf8)!
    }
    
    var finalData = Data()
    finalData.append(header.data(using: .utf8)!)
    finalData.append("\n".data(using: .utf8)!)
    finalData.append(data)
    
    try finalData.write(to: URL(fileURLWithPath: path))
}

private func generatePLYHeader(
    vertices: [PLYVertex],
    faces: [PLYFace]?,
    binary: Bool
) -> String {
    let elementVertex = "element vertex \(vertices.count)"
    let elementFace = faces.map { "element face \($0.count)" } ?? ""
    
    var properties = [
        "property float x", "property float y", "property float z"
    ]
    
    if vertices.first?.nx != nil || vertices.first?.ny != nil || vertices.first?.nz != nil {
        properties.append("property float nx")
        properties.append("property float ny")
        properties.append("property float nz")
    }
    
    if vertices.first?.r != nil || vertices.first?.g != nil || vertices.first?.b != nil {
        properties.append("property uchar red")
        properties.append("property uchar green")
        properties.append("property uchar blue")
    }

    let format = binary ? "format binary_little_endian 1.0" : "format ascii 1.0"

    var lines: [String] = []
    lines.append("ply")
    lines.append(format)
    lines.append(elementVertex)
    properties.forEach { lines.append($0) }
    if !elementFace.isEmpty {
        lines.append(elementFace)
        lines.append("property list uchar int vertex_indices")
    }
    lines.append("end_header")

    return lines.joined(separator: "\n")
}

extension Int {
    func toLittleEndian() -> Int {
        if #available(macOS 10.15, iOS 13, *) {
            return self.littleEndian
        } else {
            var value = self
            let pValue = withUnsafePointer(to: &value) { $0 }
            let bytes = Data(bytes: pValue, count: MemoryLayout<Int>.size)
            let reversed = Data(bytes.reversed())
            return reversed.withUnsafeBytes { $0.load(as: Int.self) }
        }
    }
}
