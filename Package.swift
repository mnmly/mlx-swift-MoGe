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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.17.0"),
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
        .testTarget(
            name: "MLXMoGeTests",
            dependencies: ["MLXMoGe"],
            path: "Tests/MLXMoGeTests"
        ),
    ]
)
