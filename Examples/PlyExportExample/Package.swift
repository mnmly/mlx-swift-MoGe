// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PlyExportExample",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../../"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "PlyExportExample",
            dependencies: [
                .product(name: "MLXMoGe", package: "mlx-swift-moge"),
            ]
        ),
    ]
)
