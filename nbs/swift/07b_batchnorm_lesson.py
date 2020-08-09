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

%install '.package(path: "$cwd/FastaiNotebook_06_cuda")' FastaiNotebook_06_cuda

import FastaiNotebook_06_cuda
%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

//export
import Path
import TensorFlow
import Python


# Let's start by building our own batchnorm layer from scratch. Eventually we want something like this to work:

class AlmostBatchNorm<Scalar: TensorFlowFloatingPoint> { // : Layer
    // Configuration hyperparameters
    let momentum, epsilon: Scalar
    // Running statistics
    var runningMean, runningVariance: Tensor<Scalar>
    // Trainable parameters
    var scale, offset: Tensor<Scalar>
    
    init(featureCount: Int, momentum: Scalar = 0.9, epsilon: Scalar = 1e-5) {
        (self.momentum, self.epsilon) = (momentum, epsilon)
        (scale, offset) = (Tensor(ones: [featureCount]), Tensor(zeros: [featureCount]))
        (runningMean, runningVariance) = (Tensor(0), Tensor(1))
    }

    func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let mean, variance: Tensor<Scalar>
        switch Context.local.learningPhase {
        case .training:
            mean = input.mean(alongAxes: [0, 1, 2])
            variance = input.variance(alongAxes: [0, 1, 2])
            runningMean += (mean - runningMean) * (1 - momentum)
            runningVariance += (variance - runningVariance) * (1 - momentum)
        case .inference:
            (mean, variance) = (runningMean, runningVariance)
        }
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}

# But there are some automatic differentiation limitations (lack of support for classes and control flow) that make this impossible for now, so we'll need a few workarounds. A `Reference` will let us update running statistics without making the layer a class or declaring the `applied` method `mutating`:

//export
public class Reference<T> {
    public var value: T
    public init(_ value: T) { self.value = value }
}

# The following snippet will let us differentiate a layer's `forward` method (which is the one called in `call` for `FALayer`) if it's composed of training and inference implementations that are each differentiable:

# +
//export
public protocol LearningPhaseDependent: FALayer {
    associatedtype Input
    associatedtype Output
    @differentiable func forwardTraining (_ input: Input) -> Output
    @differentiable func forwardInference(_ input: Input) -> Output
}

extension LearningPhaseDependent {
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
        case .training:  return valueWithPullback(at: input) { $0.forwardTraining($1)  }
        case .inference: return valueWithPullback(at: input) { $0.forwardInference($1) }
        }
    }
}
# -

# Now we can implement a BatchNorm that we can use in our models:

# +
//export
public protocol Norm: FALayer where Input == TF, Output == TF {
    init(_ featureCount: Int, epsilon: Float)
}

public struct FABatchNorm: LearningPhaseDependent, Norm {
    // Configuration hyperparameters
    @noDerivative var momentum, epsilon: Float
    // Running statistics
    @noDerivative let runningMean, runningVariance: Reference<TF>
    // Trainable parameters
    public var scale, offset: TF
    
    public init(_ featureCount: Int, momentum: Float, epsilon: Float = 1e-5) {
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = Tensor(ones: [featureCount])
        self.offset = Tensor(zeros: [featureCount])
        self.runningMean = Reference(Tensor(0))
        self.runningVariance = Reference(Tensor(1))
    }
    
    public init(_ featureCount: Int, epsilon: Float = 1e-5) {
        self.init(featureCount, momentum: 0.9, epsilon: epsilon)
    }

    @differentiable
    public func forwardTraining(_ input: TF) -> TF {
        let mean = input.mean(alongAxes: [0, 1, 2])
        let variance = input.variance(alongAxes: [0, 1, 2])
        runningMean.value += (mean - runningMean.value) * (1 - momentum)
        runningVariance.value += (variance - runningVariance.value) * (1 - momentum)
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
    
    @differentiable
    public func forwardInference(_ input: TF) -> TF {
        let (mean, variance) = (runningMean.value, runningVariance.value)
        let normalizer = rsqrt(variance + epsilon) * scale
        return (input - mean) * normalizer + offset
    }
}
# -

# Here is a generic `ConvNorm` layer, that combines a conv2d and a norm (like batchnorm, running batchnorm etc...) layer.

//export
public struct ConvNorm<NormType: Norm & FALayer>: FALayer
    where NormType.AllDifferentiableVariables == NormType.TangentVector {
    public var conv: FANoBiasConv2D<Float>
    public var norm: NormType
    
    public init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 2){
        self.conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu) 
        self.norm = NormType(cOut, epsilon: 1e-5)
    }

    @differentiable
    public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
        return norm(conv(input))
    }
}

//export
public struct CnnModelNormed<NormType: Norm & FALayer>: FALayer
    where NormType.AllDifferentiableVariables == NormType.TangentVector {
    public var convs: [ConvNorm<NormType>]
    public var pool = FAGlobalAvgPool2D<Float>()
    public var linear: FADense<Float>
    
    public init(channelIn: Int, nOut: Int, filters: [Int]){
        let allFilters = [channelIn] + filters
        convs = Array(0..<filters.count).map { i in
            return ConvNorm<NormType>(allFilters[i], allFilters[i+1], ks: 3, stride: 2)
        }
        linear = FADense<Float>(filters.last!, nOut)
    }
    
    @differentiable
    public func forward(_ input: TF) -> TF {
        // TODO: Work around https://bugs.swift.org/browse/TF-606
        return linear.forward(pool.forward(convs(input)))
    }
}

# Let's benchmark this batchnorm implementation!

func benchmark(forward: () -> (), backward: () -> ()) {
    print("forward:")
    time(repeating: 10, forward)
    print("backward:")
    time(repeating: 10, backward)
}

let input = TF(randomUniform: [64, 28, 28, 32])
let norm = FABatchNorm(32)
let pb = pullback(at: input) { x in norm(x) }
benchmark(forward: { norm(input) }, backward: { pb(input) })

# Yikes, that's pretty bad. Luckily, TensorFlow has a built-in fused batchnorm layer. Let's see how the performance looks for that:

let input = TF(randomUniform: [64, 28, 28, 32])
let norm = FABatchNorm(32)
let bnresult = Raw.fusedBatchNormV2(
    input, scale: norm.scale, offset: norm.offset, 
    mean: TF([] as [Float]), variance: TF([] as [Float]), 
    epsilon: Double(norm.epsilon))
benchmark(
    forward: {
        Raw.fusedBatchNormV2(
            input, scale: norm.scale, offset: norm.offset, 
            mean: TF([] as [Float]), variance: TF([] as [Float]), 
            epsilon: Double(norm.epsilon))
    },
    backward: {
        Raw.fusedBatchNormGradV2(
            yBackprop: input, input, scale: TF(norm.scale), 
            reserveSpace1: bnresult.reserveSpace1, 
            reserveSpace2: bnresult.reserveSpace2, 
            epsilon: Double(norm.epsilon))
    })

# +
struct PullbackArgs<T : TensorGroup, U : TensorGroup> : TensorGroup {
    let input: T
    let cotangent: U
}

class CompiledFunction<Input: Differentiable & TensorGroup, Output: Differentiable & TensorGroup> {
    let f: @differentiable (Input) -> Output
    init(_ f: @escaping @differentiable (Input) -> Output) {
        self.f = f
    }
}

func xlaCompiled<T : Differentiable & TensorGroup, U : Differentiable & TensorGroup>(
    _ fn: @escaping @differentiable (T) -> U) -> CompiledFunction<T, U>
    where T.TangentVector : TensorGroup, U.TangentVector : TensorGroup {
    let xlaCompiledFn: (T) -> U = _graph(fn, useXLA: true)
    let xlaCompiledPullback = _graph(
        { (pbArgs: PullbackArgs<T, U.TangentVector>) in
            pullback(at: pbArgs.input, in: fn)(pbArgs.cotangent) },
        useXLA: true
    )
    return CompiledFunction(differentiableFunction { x in
        (value: xlaCompiledFn(x), pullback: { v in
            xlaCompiledPullback(PullbackArgs(input: x, cotangent: v))})
    })
}

# +
struct TrainingKernelInput: TensorGroup, Differentiable, AdditiveArithmetic {
    var input, scale, offset, runningMean, runningVariance, momentum, epsilon: TF
}

struct TrainingKernelOutput: TensorGroup, Differentiable, AdditiveArithmetic {
    var normalized, newRunningMean, newRunningVariance: TF
}

@differentiable
func trainingKernel(_ input: TrainingKernelInput) -> TrainingKernelOutput {
    let mean = input.input.mean(alongAxes: [0, 1, 2])
    let variance = input.input.variance(alongAxes: [0, 1, 2])
    let invMomentum = TF(1) - input.momentum
    let newRunningMean = input.runningMean * input.momentum + mean * invMomentum
    let newRunningVariance = input.runningVariance * input.momentum + variance * invMomentum
    let normalizer = rsqrt(variance + input.epsilon) * input.scale
    let normalized = (input.input - mean) * normalizer + input.offset
    return TrainingKernelOutput(
        normalized: normalized,
        newRunningMean: newRunningMean,
        newRunningVariance: newRunningVariance
    )
}

# +
let input = TF(randomUniform: [64, 28, 28, 32])
let norm = FABatchNorm(32)
let compiledTrainingKernel = xlaCompiled(trainingKernel)
let kernelInput = TrainingKernelInput(
    input: input,
    scale: norm.scale,
    offset: norm.offset,
    runningMean: norm.runningMean.value,
    runningVariance: norm.runningVariance.value,
    momentum: Tensor(norm.momentum),
    epsilon: Tensor(norm.epsilon))
let pb = pullback(at: kernelInput) { x in compiledTrainingKernel.f(x) }
let kernelOutput = compiledTrainingKernel.f(kernelInput)

benchmark(
    forward: { compiledTrainingKernel.f(kernelInput) },
    backward: { pb(kernelOutput) })
# -


