// PLY Exporter for MoGe-2 point clouds.
//
// Provides convenience functions to export geometry results directly to .ply files.

import Foundation
import MLX

extension PLYVertex {
    /// Create a vertex from a 3D point with optional normal and color.
    ///
    /// - Parameters:
    ///   - position: (x, y, z) coordinates
    ///   - normal: Optional surface normal (nx, ny, nz)
    ///   - color: Optional RGB color (0-255)
    public init(
        position: (Float, Float, Float),
        normal: (Float, Float, Float)? = nil,
        color: (UInt8, UInt8, UInt8)? = nil
    ) {
        self.init(
            x: position.0, y: position.1, z: position.2,
            nx: normal?.0, ny: normal?.1, nz: normal?.2,
            r: color?.0, g: color?.1, b: color?.2
        )
    }
}

extension PLYVertex {
    /// Create a vertex from MLXArray data.
    ///
    /// - Parameters:
    ///   - xyz: (x, y, z) coordinates as Float values
    ///   - index: Index of this vertex (for face generation)
    public init(xyz: [Float], index: Int = 0) {
        self.init(
            x: xyz[0], y: xyz[1], z: xyz[2],
            nx: nil, ny: nil, nz: nil,
            r: nil, g: nil, b: nil
        )
    }
    
    /// Create a vertex from MLXArray data with normal.
    ///
    /// - Parameters:
    ///   - xyz: (x, y, z) coordinates as Float values
    ///   - nml: (nx, ny, nz) normal as Float values
    public init(xyz: [Float], nml: [Float]) {
        self.init(
            x: xyz[0], y: xyz[1], z: xyz[2],
            nx: nml[0], ny: nml[1], nz: nml[2],
            r: nil, g: nil, b: nil
        )
    }
}

extension PLYFace {
    /// Create a face from three vertex indices.
    public static func triangle(_ i0: Int, _ i1: Int, _ i2: Int) -> PLYFace {
        self.init(i0, i1, i2)
    }
    
    /// Create a quad face (will be split into two triangles).
    public static func quad(_ i0: Int, _ i1: Int, _ i2: Int, _ i3: Int) -> [PLYFace] {
        [
            PLYFace(i0, i1, i2),
            PLYFace(i0, i2, i3)
        ]
    }
}

/// Export a point cloud from depth map to PLY format.
///
/// - Parameters:
///   - path: Output file path
///   - depth: Depth array (H, W) or (B, H, W)
///   - intrinsics: Camera intrinsics matrix
///   - height: Image height
///   - width: Image width
///   - binary: Whether to write in binary format
public func exportDepthMapToPLY(
    path: String,
    depth: MLXArray,
    intrinsics: MLXArray,
    height: Int,
    width: Int,
    binary: Bool = false,
    flipYUp: Bool = false
) throws {
    let points = depthMapToPointMap(depth: depth, intrinsics: intrinsics, height: height, width: width)
    try exportPointCloudToPLY(path: path, points: points, binary: binary, flipYUp: flipYUp)
}

/// Export a point cloud to PLY format.
///
/// - Parameters:
///   - path: Output file path
///   - points: Point cloud array (H, W, 3) or (B, H, W, 3)
///   - binary: Whether to write in binary format
/// Convert a point cloud (and optional colors) into `[PLYVertex]` without writing to disk.
///
/// - Parameters:
///   - points: Point cloud array shaped `(H, W, 3)`, `(B, H, W, 3)`, or `(N, 3)`.
///   - colors: Optional color array with a matching layout, values in `[0, 255]`.
///   - flipYUp: If true, negates Y and Z to convert MoGe's OpenCV convention
///     (+Y down, +Z forward) into +Y-up / +Z-back used by Blender/GL viewers.
public func pointCloudToVertices(
    points: MLXArray,
    colors: MLXArray? = nil,
    flipYUp: Bool = false
) -> [PLYVertex] {
    let pointData = points.asType(.float32).asArray(Float.self)
    let colorData = colors?.asType(.float32).asArray(Float.self)
    if let colorData { precondition(colorData.count == pointData.count, "colors must match points layout") }

    let count = pointData.count / 3
    let sign: Float = flipYUp ? -1 : 1
    var vertices: [PLYVertex] = []
    vertices.reserveCapacity(count)

    for i in 0..<count {
        let idx = i * 3
        var color: (UInt8, UInt8, UInt8)? = nil
        if let colorData {
            let r = UInt8(min(max(colorData[idx], 0), 255))
            let g = UInt8(min(max(colorData[idx + 1], 0), 255))
            let b = UInt8(min(max(colorData[idx + 2], 0), 255))
            color = (r, g, b)
        }
        vertices.append(
            PLYVertex(
                position: (pointData[idx], sign * pointData[idx + 1], sign * pointData[idx + 2]),
                normal: nil,
                color: color
            )
        )
    }
    return vertices
}

public func exportPointCloudToPLY(
    path: String,
    points: MLXArray,
    binary: Bool = true,
    flipYUp: Bool = false
) throws {
    let vertices = pointCloudToVertices(points: points, colors: nil, flipYUp: flipYUp)
    try exportPLY(to: path, vertices: vertices, faces: nil, binary: binary)
}

/// Export a point cloud with colors to PLY format.
///
/// - Parameters:
///   - path: Output file path
///   - points: Point cloud array (H, W, 3) or (B, H, W, 3)
///   - colors: Color array (H, W, 3) or (B, H, W, 3), values in [0, 255]
///   - binary: Whether to write in binary format
public func exportPointCloudWithColorsToPLY(
    path: String,
    points: MLXArray,
    colors: MLXArray,
    binary: Bool = true,
    flipYUp: Bool = false
) throws {
    let vertices = pointCloudToVertices(points: points, colors: colors, flipYUp: flipYUp)
    
    try exportPLY(to: path, vertices: vertices, faces: nil, binary: binary)
}

/// Export a mesh to PLY format.
///
/// - Parameters:
///   - path: Output file path
///   - vertices: Array of vertex positions (x, y, z)
///   - faces: Array of triangular face indices
///   - binary: Whether to write in binary format
public func exportMeshToPLY(
    path: String,
    vertices: [(Float, Float, Float)],
    faces: [(Int, Int, Int)],
    binary: Bool = true
) throws {
    let plyVertices = vertices.map { PLYVertex(position: $0) }
    let plyFaces = faces.map { PLYFace($0.0, $0.1, $0.2) }
    try exportPLY(to: path, vertices: plyVertices, faces: plyFaces, binary: binary)
}

/// Export a mesh with vertex normals to PLY format.
///
/// - Parameters:
///   - path: Output file path
///   - vertices: Array of vertex positions (x, y, z)
///   - normals: Array of vertex normals (nx, ny, nz)
///   - faces: Array of triangular face indices
///   - binary: Whether to write in binary format
public func exportMeshWithNormalsToPLY(
    path: String,
    vertices: [(Float, Float, Float)],
    normals: [(Float, Float, Float)],
    faces: [(Int, Int, Int)],
    binary: Bool = true
) throws {
    precondition(vertices.count == normals.count, "Vertices and normals must have same count")
    
    let plyVertices = zip(vertices, normals).map { PLYVertex(position: $0.0, normal: $0.1) }
    let plyFaces = faces.map { PLYFace($0.0, $0.1, $0.2) }
    try exportPLY(to: path, vertices: plyVertices, faces: plyFaces, binary: binary)
}
