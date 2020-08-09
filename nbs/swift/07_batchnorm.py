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
%install '.package(path: "$cwd/FastaiNotebook_06_cuda")' FastaiNotebook_06_cuda

//export
import Path
import TensorFlow
import Python

import FastaiNotebook_06_cuda

%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

# ## Load data

let data = mnistDataBunch(flat: false, bs: 512)

func optFunc(_ model: CnnModel) -> SGD<CnnModel> { return SGD(for: model, learningRate: 0.4) }
func modelInit() -> CnnModel { return CnnModel(channelIn: 1, nOut: 10, filters: [8, 16, 32, 32]) }
let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.addDelegates([learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std),
                      learner.makeAddChannel()])

time { try! learner.fit(1) }

# ## Batchnorm

# ### Custom

# Let's start by building our own `BatchNorm` layer from scratch. Eventually we intend for this code to do the trick:

struct AlmostBatchNorm<Scalar: TensorFlowFloatingPoint>: Differentiable {
    // Configuration hyperparameters
    @noDerivative let momentum, epsilon: Scalar
    // Running statistics
    @noDerivative var runningMean, runningVariance: Tensor<Scalar>
    // Trainable parameters
    var scale, offset: Tensor<Scalar>
    
    init(featureCount: Int, momentum: Scalar = 0.9, epsilon: Scalar = 1e-5) {
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
        self.runningMean = Tensor(0)
        self.runningVariance = Tensor(1)
    }

    mutating func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean: Tensor<Scalar>
        let variance: Tensor<Scalar>
        switch Context.local.learningPhase {
        case .training:
            mean = input.mean(alongAxes: [0, 1, 2])
            variance = input.variance(alongAxes: [0, 1, 2])
            runningMean += (mean - runningMean) * (1 - momentum)
            runningVariance += (variance - runningVariance) * (1 - momentum)
        case .inference:
            mean = runningMean
            variance = runningVariance
        }
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}

# But there are some automatic differentiation limitations (control flow support) and `Layer` protocol constraints (mutating `call`) that make this impossible for now (note the lack of `@differentiable` or a `Layer` conformance), so we'll need a few workarounds. A `Reference` will let us update running statistics without declaring the `applied` method `mutating`:

//export
public class Reference<T> {
    public var value: T
    public init(_ value: T) { self.value = value }
}

# The following snippet will let us differentiate a layer's `call` method if it's composed of training and inference implementations that are each differentiable:

# +
//export
public protocol LearningPhaseDependent: FALayer {
    associatedtype Input
    associatedtype Output
    
    @differentiable func forwardTraining(_ input: Input) -> Output
    @differentiable func forwardInference(_ input: Input) -> Output
}

extension LearningPhaseDependent {
    // This `@differentiable` attribute is necessary, to tell the compiler that this satisfies the FALayer
    // protocol requirement, even though there is a `@differentiating(forward)` method below.
    // TODO: It seems nondeterministically necessary. Some subsequent notebooks import this successfully without it,
    // some require it. Investigate.
    @differentiable
    public func forward(_ input: Input) -> Output {
        switch Context.local.learningPhase {
        case .training:  return forwardTraining(input)
        case .inference: return forwardInference(input)
        }
    }

    @differentiating(forward)
    func gradForward(_ input: Input) ->
        (value: Output, pullback: (Self.Output.TangentVector) ->
            (Self.TangentVector, Self.Input.TangentVector)) {
        switch Context.local.learningPhase {
        case .training:
            return valueWithPullback(at: input) { $0.forwardTraining ($1) }
        case .inference:
            return valueWithPullback(at: input) { $0.forwardInference($1) }
        }
    }
}
# -

# Now we can implement a BatchNorm that we can use in our models:

# +
//export
public protocol Norm: Layer where Input == Tensor<Scalar>, Output == Tensor<Scalar>{
    associatedtype Scalar
    init(featureCount: Int, epsilon: Scalar)
}

public struct FABatchNorm<Scalar: TensorFlowFloatingPoint>: LearningPhaseDependent, Norm {
    // TF-603 workaround.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    
    // Configuration hyperparameters
    @noDerivative var momentum, epsilon: Scalar
    // Running statistics
    @noDerivative let runningMean, runningVariance: Reference<Tensor<Scalar>>
    // Trainable parameters
    public var scale, offset: Tensor<Scalar>
    
    public init(featureCount: Int, momentum: Scalar, epsilon: Scalar = 1e-5) {
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
        self.runningMean = Reference(Tensor(0))
        self.runningVariance = Reference(Tensor(1))
    }
    
    public init(featureCount: Int, epsilon: Scalar = 1e-5) {
        self.init(featureCount: featureCount, momentum: 0.9, epsilon: epsilon)
    }

    @differentiable
    public func forwardTraining(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: [0, 1, 2])
        let variance = input.variance(alongAxes: [0, 1, 2])
        runningMean.value += (mean - runningMean.value) * (1 - momentum)
        runningVariance.value += (variance - runningVariance.value) * (1 - momentum)
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
    
    @differentiable
    public func forwardInference(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = runningMean.value
        let variance = runningVariance.value
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}
# -

# TensorFlow provides a highly optimized batch norm implementation, let us redefine our batch norm to invoke it directly. 

# +
//export
struct BatchNormResult<Scalar : TensorFlowFloatingPoint> : Differentiable{
    var y, batchMean, batchVariance, reserveSpace1, reserveSpace2: Tensor<Scalar>
}

public struct TFBatchNorm<Scalar: TensorFlowFloatingPoint>: LearningPhaseDependent, Norm {
    // Configuration hyperparameters
    @noDerivative var momentum, epsilon: Scalar
    // Running statistics
    @noDerivative let runningMean, runningVariance: Reference<Tensor<Scalar>>
    // Trainable parameters
    public var scale, offset: Tensor<Scalar>
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    
    public init(featureCount: Int, momentum: Scalar, epsilon: Scalar = 1e-5) {
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
        self.runningMean = Reference(Tensor(0))
        self.runningVariance = Reference(Tensor(1))
    }
    
    public init(featureCount: Int, epsilon: Scalar = 1e-5) {
        self.init(featureCount: featureCount, momentum: 0.9, epsilon: epsilon)
    }

    @differentiable
    public func forwardTraining(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let res = TFBatchNorm<Scalar>.fusedBatchNorm(
            input, scale: scale, offset: offset, epsilon: epsilon)
        let (output, mean, variance) = (res.y, res.batchMean, res.batchVariance)
        runningMean.value += (mean - runningMean.value) * (1 - momentum)
        runningVariance.value += (variance - runningVariance.value) * (1 - momentum)
        return output
     }
    
    @differentiable
    public func forwardInference(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = runningMean.value
        let variance = runningVariance.value
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
    
    @differentiable(wrt: (x, scale, offset), vjp: _vjpFusedBatchNorm)
    static func fusedBatchNorm(
        _ x : Tensor<Scalar>, scale: Tensor<Scalar>, offset: Tensor<Scalar>, epsilon: Scalar
    ) -> BatchNormResult<Scalar> {
        let ret = Raw.fusedBatchNormV2(
            x, scale: scale, offset: offset, 
            mean: Tensor<Scalar>([] as [Scalar]), variance: Tensor<Scalar>([] as [Scalar]), 
            epsilon: Double(epsilon))
        return BatchNormResult(
            y: ret.y, batchMean: ret.batchMean, batchVariance: ret.batchVariance,
            reserveSpace1: ret.reserveSpace1, reserveSpace2: ret.reserveSpace2
        )
    }

    static func _vjpFusedBatchNorm(
        _ x : Tensor<Scalar>, scale: Tensor<Scalar>, offset: Tensor<Scalar>, epsilon: Scalar
    ) -> (BatchNormResult<Scalar>, 
          (BatchNormResult<Scalar>.TangentVector) -> (Tensor<Scalar>.TangentVector, 
                                                        Tensor<Scalar>.TangentVector, 
                                                        Tensor<Scalar>.TangentVector)) {
      let bnresult = fusedBatchNorm(x, scale: scale, offset: offset, epsilon: epsilon)
  
        return (
            bnresult, 
            {v in 
                let res = Raw.fusedBatchNormGradV2(
                    yBackprop: v.y, x, scale: Tensor<Float>(scale), 
                    reserveSpace1: bnresult.reserveSpace1, 
                    reserveSpace2: bnresult.reserveSpace2, 
                    epsilon: Double(epsilon))
                return (res.xBackprop, res.scaleBackprop, res.offsetBackprop)
            })
    }
}
# -

//export
public struct ConvBN<Scalar: TensorFlowFloatingPoint>: FALayer {
    // TF-603 workaround.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    public var conv: FANoBiasConv2D<Scalar>
    public var norm: FABatchNorm<Scalar>
    
    public init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 1){
        // TODO (when control flow AD works): use Conv2D without bias
        self.conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu)
        self.norm = FABatchNorm(featureCount: cOut, epsilon: 1e-5)
    }

    @differentiable
    public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return norm.forward(conv.forward(input))
    }
}

# +
// Would be great if this generic could work
// struct ConvNorm<NormType: Norm, Scalar: TensorFlowFloatingPoint>: Layer
//     where NormType.Scalar == Scalar {
//     var conv: Conv2D<Scalar>
//     var norm: NormType
//     init(
//         filterShape: (Int, Int, Int, Int),
//         strides: (Int, Int) = (1, 1),
//         padding: Padding = .valid,
//         activation: @escaping Conv2D<Scalar>.Activation = identity
//     ) {
//         // TODO (when control flow AD works): use Conv2D without bias
//         self.conv = Conv2D(
//             filterShape: filterShape,
//             strides: strides,
//             padding: padding,
//             activation: activation)
//         self.norm = NormType.init(featureCount: filterShape.3, epsilon: 1e-5)
//     }

//     @differentiable
//     func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
//         return norm.applied(to: conv.applied(to: input))
//     }
// }
//typealias ConvBN = ConvNorm<BatchNorm<Float>, Float>
# -

//export
public struct CnnModelBN: Layer {
    public var convs: [ConvBN<Float>]
    public var pool = FAGlobalAvgPool2D<Float>()
    public var linear: FADense<Float>
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    
    public init(channelIn: Int, nOut: Int, filters: [Int]){
        let allFilters = [channelIn] + filters
        convs = Array(0..<filters.count).map { i in
            return ConvBN(allFilters[i], allFilters[i+1], ks: 3, stride: 2)
        }
        linear = FADense<Float>(filters.last!, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        // TODO: Work around https://bugs.swift.org/browse/TF-606
        return linear.forward(pool.forward(convs(input)))
    }
}

func optFunc(_ model: CnnModelBN) -> SGD<CnnModelBN> { return SGD(for: model, learningRate: 0.4) }
func modelInit() -> CnnModelBN { return CnnModelBN(channelIn: 1, nOut: 10, filters: [8, 16, 32, 32]) }
let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.addDelegates([learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std),
                      learner.makeAddChannel()])

time { try! learner.fit(1) }

# ## More norms

# ### Layer norm

# From [the paper](https://arxiv.org/abs/1607.06450): "*batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the minibatches have to be small*".

# General equation for a norm layer with learnable affine:
#
# $$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$
#
# The difference with BatchNorm is
# 1. we don't keep a moving average
# 2. we don't average over the batches dimension but over the hidden dimension, so it's independent of the batch size

# +
struct LayerNorm2D<Scalar: TensorFlowFloatingPoint>: Norm {
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    // Configuration hyperparameters
    @noDerivative let epsilon: Scalar
    // Trainable parameters
    var scale: Tensor<Scalar>
    var offset: Tensor<Scalar>
    
    init(featureCount: Int, epsilon: Scalar = 1e-5) {
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: [1, 2, 3])
        let variance = input.variance(alongAxes: [1, 2, 3])
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}

struct ConvLN<Scalar: TensorFlowFloatingPoint>: FALayer {
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    var conv: FANoBiasConv2D<Scalar>
    var norm: LayerNorm2D<Scalar>
    
    init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 2){
        // TODO (when control flow AD works): use Conv2D without bias
        self.conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu)
        self.norm = LayerNorm2D(featureCount: cOut, epsilon: 1e-5)
    }

    @differentiable
    func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return norm.callAsFunction(conv.forward(input))
    }
}
# -

public struct CnnModelLN: Layer {
    public var convs: [ConvLN<Float>]
    public var pool = FAGlobalAvgPool2D<Float>()
    public var linear: FADense<Float>
    
    public init(channelIn: Int, nOut: Int, filters: [Int]){
        let allFilters = [channelIn] + filters
        convs = Array(0..<filters.count).map { i in
            return ConvLN(allFilters[i], allFilters[i+1], ks: 3, stride: 2)
        }
        linear = FADense<Float>(filters.last!, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        // TODO: Work around https://bugs.swift.org/browse/TF-606
        return linear.forward(pool.forward(convs(input)))
    }
}


# +
struct InstanceNorm<Scalar: TensorFlowFloatingPoint>: Norm {
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    // Configuration hyperparameters
    @noDerivative let epsilon: Scalar
    // Trainable parameters
    var scale: Tensor<Scalar>
    var offset: Tensor<Scalar>
    
    init(featureCount: Int, epsilon: Scalar = 1e-5) {
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = input.mean(alongAxes: [2, 3])
        let variance = input.variance(alongAxes: [2, 3])
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}

struct ConvIN<Scalar: TensorFlowFloatingPoint>: FALayer {
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    var conv: FANoBiasConv2D<Scalar>
    var norm: InstanceNorm<Scalar>
    
    init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 2){
        // TODO (when control flow AD works): use Conv2D without bias
        self.conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu)
        self.norm = InstanceNorm(featureCount: cOut, epsilon: 1e-5)
    }

    @differentiable
    func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return norm.callAsFunction(conv.forward(input))
    }
}
# -

# Lost in all those norms? The authors from the [group norm paper](https://arxiv.org/pdf/1803.08494.pdf) have you covered:
#
# ![Various norms](../dev_course/dl2/images/norms.png)

# TODO/skipping GroupNorm

# ### Running Batch Norm

struct RunningBatchNorm<Scalar: TensorFlowFloatingPoint>: LearningPhaseDependent, Norm {
    @noDerivative public var delegates: [(Self.Output) -> ()] = []
    // Configuration hyperparameters
    @noDerivative let momentum: Scalar
    @noDerivative let epsilon: Scalar
    // Running statistics
    @noDerivative let runningSum: Reference<Tensor<Scalar>>
    @noDerivative let runningSumOfSquares: Reference<Tensor<Scalar>>
    @noDerivative let runningCount: Reference<Scalar>
    @noDerivative let samplesSeen: Reference<Int>
    // Trainable parameters
    var scale: Tensor<Scalar>
    var offset: Tensor<Scalar>
    
    init(featureCount: Int, momentum: Scalar, epsilon: Scalar = 1e-5) {
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
        self.runningSum = Reference(Tensor(0))
        self.runningSumOfSquares = Reference(Tensor(0))
        self.runningCount = Reference(Scalar(0))
        self.samplesSeen = Reference(0)
    }
    
    init(featureCount: Int, epsilon: Scalar = 1e-5) {
        self.init(featureCount: featureCount, momentum: 0.9, epsilon: epsilon)
    }

    @differentiable
    func forwardTraining(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let (batch, channels) = (input.shape[0], Scalar(input.shape[3]))
        let sum = input.sum(alongAxes: [0, 1, 2])
        let sumOfSquares = (input * input).sum(alongAxes: [0, 1, 2])
        // TODO: Work around https://bugs.swift.org/browse/TF-607
        let count = withoutDerivative(at: Scalar(input.scalarCount)) { tmp in tmp } / channels
        let mom = momentum / sqrt(Scalar(batch) - 1)
        let runningSum = mom * self.runningSum.value + (1 - mom) * sum
        let runningSumOfSquares = mom * self.runningSumOfSquares.value + (
            1 - mom) * sumOfSquares
        let runningCount = mom * self.runningCount.value + (1 - mom) * count
        
        self.runningSum.value = runningSum
        self.runningSumOfSquares.value = runningSumOfSquares
        self.runningCount.value = runningCount
        self.samplesSeen.value += batch
        
        let mean = runningSum / runningCount
        let variance = runningSumOfSquares / runningCount - mean * mean
        
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
    
    @differentiable
    func forwardInference(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean = runningSum.value / runningCount.value
        let variance = runningSumOfSquares.value / runningCount.value - mean * mean
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}

# TODO: XLA compilation + test RBN

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"07_batchnorm.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))




