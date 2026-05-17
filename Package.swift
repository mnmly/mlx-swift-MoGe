// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mlx-swift-moge",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MLXMoGe",
            targets: ["MLXMoGe"]
        ),
        .executable(name: "moge-bench", targets: ["moge-bench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "MLXMoGe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/MLXMoGe"
        ),
        .executableTarget(
            name: "moge-bench",
            dependencies: [
                "MLXMoGe",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/moge-bench"
        ),
        .testTarget(
            name: "MLXMoGeTests",
            dependencies: ["MLXMoGe"],
            path: "Tests/MLXMoGeTests"
        ),
    ]
)
