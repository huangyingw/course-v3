# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Swift
#     language: swift
#     name: swift
# ---

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_02_fully_connected")' FastaiNotebook_02_fully_connected

//export
import Foundation
import TensorFlow
import Path

import FastaiNotebook_02_fully_connected

# ## Does nn.Conv2d init work well?

var (xTrain, yTrain, xValid, yValid) = loadMNIST(path: Path.home/".fastai"/"data"/"mnist_tst")
let (trainMean, trainStd) = (xTrain.mean(), xTrain.standardDeviation())
xTrain = normalize(xTrain, mean: trainMean, std: trainStd)
xValid = normalize(xValid, mean: trainMean, std: trainStd)

xTrain = xTrain.reshaped(to: [xTrain.shape[0], 28, 28, 1])
xValid = xValid.reshaped(to: [xValid.shape[0], 28, 28, 1])
print(xTrain.shape, xValid.shape)

let images = xTrain.shape[0]
let classes = xValid.max() + 1
let channels = 32

var layer1 = FAConv2D<Float>(filterShape: (5, 5, 1, channels)) //Conv2D(1, nh, 5)

let x = xValid[0..<100]

x.shape

extension Tensor where Scalar: TensorFlowFloatingPoint {
    func stats() -> (mean: Tensor, std: Tensor) {
        return (mean: mean(), std: standardDeviation())
    }
}

(filter: layer1.filter.stats(), bias: layer1.bias.stats())

withDevice(.cpu){
    let result = layer1(x)
}

let result = layer1(x)

result.stats()

# This is in 1a now so this code is disabled from here:
#
# ```swift
# var rng = PhiloxRandomNumberGenerator.global
#
# extension Tensor where Scalar: TensorFlowFloatingPoint {
#     init(kaimingNormal shape: TensorShape, negativeSlope: Double = 1.0) {
#         // Assumes Leaky ReLU nonlinearity
#         let gain = Scalar(sqrt(2.0 / (1.0 + pow(negativeSlope, 2))))
#         let spatialDimCount = shape.count - 2
#         let receptiveField = shape[0..<spatialDimCount].contiguousSize
#         let fanIn = shape[shape.count - 2] * receptiveField
#         self.init(randomNormal: shape,
#                   stddev: gain / sqrt(Scalar(fanIn)),
#                   generator: &rng
#         )
#     }
# }
# ```

layer1.filter = Tensor(kaimingNormal: layer1.filter.shape, negativeSlope: 1.0)
layer1(x).stats()

// export
func leakyRelu<T: TensorFlowFloatingPoint>(
    _ x: Tensor<T>,
    negativeSlope: Double = 0.0
) -> Tensor<T> {
    return max(0, x) + T(negativeSlope) * min(0, x)
}

layer1.filter = Tensor(kaimingNormal: layer1.filter.shape, negativeSlope: 0.0)
leakyRelu(layer1(x)).stats()

var layer1 = FAConv2D<Float>(filterShape: (5, 5, 1, channels)) //Conv2D(1, nh, 5)
leakyRelu(layer1(x)).stats()

layer1.filter.shape

let spatialDimCount = layer1.filter.rank - 2
let receptiveField = layer1.filter.shape[0..<spatialDimCount].contiguousSize
receptiveField

let filtersIn = layer1.filter.shape[2]
let filtersOut = layer1.filter.shape[3]
print(filtersIn, filtersOut)

let fanIn = filtersIn * receptiveField
let fanOut = filtersOut * receptiveField
print(fanIn, fanOut)

func gain(_ negativeSlope: Double) -> Double {
    return sqrt(2.0 / (1.0 + pow(negativeSlope, 2.0)))
}

(gain(1.0), gain(0.0), gain(0.01), gain(0.1), gain(sqrt(5.0)))

(2 * Tensor<Float>(randomUniform: [10000]) - 1).standardDeviation()

1.0 / sqrt(3.0)

//export
extension Tensor where Scalar: TensorFlowFloatingPoint {
    init(kaimingUniform shape: TensorShape, negativeSlope: Double = 1.0) {
        // Assumes Leaky ReLU nonlinearity
        let gain = Scalar.init(TensorFlow.sqrt(2.0 / (1.0 + TensorFlow.pow(negativeSlope, 2))))
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[shape.count - 2] * receptiveField
        let bound = TensorFlow.sqrt(Scalar(3.0)) * gain / TensorFlow.sqrt(Scalar(fanIn))
        self = bound * (2 * Tensor(randomUniform: shape, generator: &PhiloxRandomNumberGenerator.global) - 1)
    }
}

layer1.filter = Tensor(kaimingUniform: layer1.filter.shape, negativeSlope: 0.0)
leakyRelu(layer1(x)).stats()

layer1.filter = Tensor(kaimingUniform: layer1.filter.shape, negativeSlope: sqrt(5.0))
leakyRelu(layer1(x)).stats()

public struct Model: Layer {
    public var conv1 = FAConv2D<Float>(
        filterShape: (5, 5, 1, 8),   strides: (2, 2), padding: .same, activation: relu
    )
    public var conv2 = FAConv2D<Float>(
        filterShape: (3, 3, 8, 16),  strides: (2, 2), padding: .same, activation: relu
    )
    public var conv3 = FAConv2D<Float>(
        filterShape: (3, 3, 16, 32), strides: (2, 2), padding: .same, activation: relu
    )
    public var conv4 = FAConv2D<Float>(
        filterShape: (3, 3, 32, 1),  strides: (2, 2), padding: .valid
    )
    public var flatten = Flatten<Float>()

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv1, conv2, conv3, conv4, flatten)
    }
}

let y = Tensor<Float>(yValid[0..<100])
var model = Model()

let prediction = model(x)
prediction.stats()

# +
let gradients = gradient(at: model) { model in
    meanSquaredError(predicted: model(x), expected: y)
}

gradients.conv1.filter.stats()
# -

for keyPath in [\Model.conv1, \Model.conv2, \Model.conv3, \Model.conv4] {
    model[keyPath: keyPath].filter = Tensor(kaimingUniform: model[keyPath: keyPath].filter.shape)
}

let prediction = model(x)
prediction.stats()

# +
let gradients = gradient(at: model) { model in
    meanSquaredError(predicted: model(x), expected: y)
}

gradients.conv1.filter.stats()
# -

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"02a_why_sqrt5.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


