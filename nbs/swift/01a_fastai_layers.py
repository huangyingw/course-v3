# -*- coding: utf-8 -*-
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

# # "fast.ai" layers
#
# This notebook defines layers similar to those defined in the [Swift for TensorFlow Deep Learning Library](https://github.com/tensorflow/swift-apis), but with some experimental extra features for the fast.ai course.

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_01_matmul")' FastaiNotebook_01_matmul

//export
import Path
import TensorFlow

import FastaiNotebook_01_matmul

# ## Tensor extensions

# All the fastai layers will use kaiming initialization as default, so we write an extension to do this fast.

// export
public extension Tensor where Scalar: TensorFlowFloatingPoint {
    init(kaimingNormal shape: TensorShape, negativeSlope: Double = 1.0) {
        // Assumes Leaky ReLU nonlinearity
        let gain = Scalar.init(TensorFlow.sqrt(2.0 / (1.0 + TensorFlow.pow(negativeSlope, 2))))
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[spatialDimCount] * receptiveField
        self.init(randomNormal: shape)
        self *= Tensor<Scalar>(gain/TensorFlow.sqrt(Scalar(fanIn)))
    }
}

# We shorten the name of standardDeviation to match numpy/PyTorch since we'll have to write it a lot in the notebooks.

// export
public extension Tensor where Scalar: TensorFlowFloatingPoint {
    func std() -> Tensor<Scalar> { return standardDeviation() }
    func std(alongAxes a: [Int]) -> Tensor<Scalar> { return standardDeviation(alongAxes: a) }
    func std(alongAxes a: Tensor<Int32>) -> Tensor<Scalar> { return standardDeviation(alongAxes: a) }
    func std(alongAxes a: Int...) -> Tensor<Scalar> { return standardDeviation(alongAxes: a) }
    func std(squeezingAxes a: [Int]) -> Tensor<Scalar> { return standardDeviation(squeezingAxes: a) }
    func std(squeezingAxes a: Tensor<Int32>) -> Tensor<Scalar> { return standardDeviation(squeezingAxes: a) }
    func std(squeezingAxes a: Int...) -> Tensor<Scalar> { return standardDeviation(squeezingAxes: a) }
}

# # Introducing Protocols
#
# Just like PyTorch, the S4TF base library provides a standard layer type.  It is defined in the [TensorFlow/swift-apis](https://github.com/tensorflow/swift-apis/) repository on GitHub in [Layer.swift](https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layer.swift) and it looks like this:
#
# ```swift
# protocol Layer: Differentiable {
#     /// The input type of the layer.
#     associatedtype Input: Differentiable
#     /// The output type of the layer.
#     associatedtype Output: Differentiable
#
#     /// Returns the output obtained from applying the layer to the given input.
#     ///
#     /// - Parameter input: The input to the layer.
#     /// - Returns: The output.
#     @differentiable
#     func call(_ input: Input) -> Output
# }
# ```
#
# There is a lot going on here - before we dive into the details of Layers, lets talk about what protocols are:
#
# **Slides**: [Protocols in Swift](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g5674d3ead7_0_474)
#

# # Explaining `Layer`
#  
# Let's break down the `Layer` protocol into its pieces:
#
# ```swift
# protocol Layer: Differentiable {
#     ...
# }
# ```
#
# This declares a protocol named `Layer` and says that all layers are also `Differentiable`.  `Differentiable` is a yet-another protocol [defined in the Swift standard library](https://github.com/apple/swift/blob/tensorflow/stdlib/public/core/AutoDiff.swift#L99).
#
# ```swift
#     /// Returns the output obtained from applying the layer to the given input.
#     ///
#     /// - Parameter input: The input to the layer.
#     /// - Returns: The output.
#     @differentiable
#     func call(_ input: Input) -> Output
# ```
#
# These lines say that all `Layer`s are required to have a method named `call`, that it must be differentiable, and that it can take an return an arbitrary `Input` and `Output` type.
#
# ```swift
#     /// The input type of the layer.
#     associatedtype Input: Differentiable
#     /// The output type of the layer.
#     associatedtype Output: Differentiable
# ```
#
# These two lines say that each `Layer` has to have an `Input` and `Output` type, and that those types must furthermore be differentiable.
#

# # Taking `Layer` further: `FALayer`
#
# Jeremy likes to have nice things, and in particular wants to be able to install delegate callbacks on `Layer`s (to mimic PyTorch's [hooks](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks)).  It might seem that we have to give up on using the builtin layer, but that isn't so!  We can extend the existing `Layer` and give it new functionality with `FALayer`.  First we define the type we want to use for our callbacks:

# We can now define `FALayer`, which is just a layer that has a delegate.  Types that implement `FALayer` will implement a new `forward` method instead of `func call`:

# +
//export

// FALayer is a layer that supports callbacks through its LayerDelegate.
public protocol FALayer: Layer {
    var delegates: [(Output) -> ()] { get set }
    
    // FALayer's will implement this instead of `func call`.
    @differentiable
    func forward(_ input: Input) -> Output
    
    associatedtype Input
    associatedtype Output
}
# -

# It would be a tremendous amount of boilerplate to require all `FALayer`s to implement `func call` and invoke their delegate, so we'll do that once for all `FALayer`s right here in an extension.  We implement it by calling the `forward` function and then invoke the delegate:

//export
public extension FALayer {
    @differentiable(vjp: callGrad)
    func callAsFunction(_ input: Input) -> Output {
        let activation = forward(input)
        for d in delegates { d(activation) }
        return activation
    }
       
    // NOTE: AutoDiff synthesizes a leaking VJP for this, so we define a custom VJP.
    //    TF-475: https://bugs.swift.org/browse/TF-475
    // NOTE: If we use `@differentiating`, then there is a linker error. So we use `@differentiable` instead.
    //    TF-476: https://bugs.swift.org/browse/TF-476
    func callGrad(_ input: Input) ->
        (Output, (Self.Output.TangentVector) -> (Self.TangentVector, Self.Input.TangentVector)) {
        return Swift.valueWithPullback(at: self, input) { (m, i) in m.forward(i) }
    }
    
    //We also add a default init to our `delegates` variable, so that we don't have to define it each time, as
    //well as a function to easily add a delegate.
    //var delegates: [(Output) -> ()] { 
    //    get { return [] }
    //    set {}
    //}
    
    mutating func addDelegate(_ d: @escaping (Output) -> ()) { delegates.append(d) }
}


# Now we can start implementing some layers, like this dense layer:

# # Dense

# The code is copy-pasted from S4TF for the most general init, and we add a convenience init with kaiming initialization for the weights.

# +
//export

@frozen
public struct FADense<Scalar: TensorFlowFloatingPoint>: FALayer {
    // Note: remove the explicit typealiases after TF-603 is resolved.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    @noDerivative public var delegates: [(Output) -> ()] = []
    @noDerivative public let activation: Activation

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation
    ) {
        self.weight = weight
        self.bias = bias
        self.activation = activation
    }

    @differentiable
    public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(input • weight + bias)
    }
}

public extension FADense {
    init(_ nIn: Int, _ nOut: Int, activation: @escaping Activation = identity) {
        self.init(weight: Tensor(kaimingNormal: [nIn, nOut], negativeSlope: 1.0),
                  bias: Tensor(zeros: [nOut]),
                  activation: activation)
    }
}
# -

# # Conv2D

# +
//export

@frozen
public struct FANoBiasConv2D<Scalar: TensorFlowFloatingPoint>: FALayer {
    // TF-603 workaround.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    
    public var filter: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    @noDerivative public let activation: Activation
    @noDerivative public let strides: (Int, Int)
    @noDerivative public let padding: Padding
    @noDerivative public var delegates: [(Output) -> ()] = []

    public init(
        filter: Tensor<Scalar>,
        activation: @escaping Activation,
        strides: (Int, Int),
        padding: Padding
    ) {
        self.filter = filter
        self.activation = activation
        self.strides = strides
        self.padding = padding
    }

    @differentiable
    public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(conv2D(input, filter: filter,
                                        strides: (1, strides.0, strides.1, 1),
                                        padding: padding))
    }
}

public extension FANoBiasConv2D {
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .same,
        activation: @escaping Activation = identity
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1,
            filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(kaimingNormal: filterTensorShape, negativeSlope: 1.0),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

public extension FANoBiasConv2D {
    init(_ cIn: Int, _ cOut: Int, ks: Int, stride: Int = 1, padding: Padding = .same,
         activation: @escaping Activation = identity){
        self.init(filterShape: (ks, ks, cIn, cOut),
                  strides: (stride, stride),
                  padding: padding,
                  activation: activation)
    }
}

# +
//export

@frozen
public struct FAConv2D<Scalar: TensorFlowFloatingPoint>: FALayer {
    // Note: remove the explicit typealiases after TF-603 is resolved.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    
    public var filter: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    @noDerivative public let activation: Activation
    @noDerivative public let strides: (Int, Int)
    @noDerivative public let padding: Padding
    @noDerivative public var delegates: [(Output) -> ()] = []

    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation,
        strides: (Int, Int),
        padding: Padding
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
    }

    @differentiable
    public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(conv2D(input, filter: filter,
                                        strides: (1, strides.0, strides.1, 1),
                                        padding: padding) + bias)
    }
}

public extension FAConv2D {
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .same,
        activation: @escaping Activation = identity
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1,
            filterShape.2, filterShape.3])
        self.init(
            filter: Tensor(kaimingNormal: filterTensorShape, negativeSlope: 1.0),
            bias: Tensor(zeros: TensorShape([filterShape.3])),
            activation: activation,
            strides: strides,
            padding: padding)
    }
}

public extension FAConv2D {
    init(_ cIn: Int, _ cOut: Int, ks: Int, stride: Int = 1, padding: Padding = .same,
         activation: @escaping Activation = identity){
        self.init(filterShape: (ks, ks, cIn, cOut),
                  strides: (stride, stride),
                  padding: padding,
                  activation: activation)
    }
}
# -

# # AvgPool2D

# +
//export

@frozen
public struct FAAvgPool2D<Scalar: TensorFlowFloatingPoint>: FALayer {
    // TF-603 workaround.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    
    @noDerivative let poolSize: (Int, Int, Int, Int)
    @noDerivative let strides: (Int, Int, Int, Int)
    @noDerivative let padding: Padding
    @noDerivative public var delegates: [(Output) -> ()] = []

    public init(
        poolSize: (Int, Int, Int, Int),
        strides: (Int, Int, Int, Int),
        padding: Padding
    ) {
        self.poolSize = poolSize
        self.strides = strides
        self.padding = padding
    }

    public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
        self.poolSize = (1, poolSize.0, poolSize.1, 1)
        self.strides = (1, strides.0, strides.1, 1)
        self.padding = padding
    }
    
    public init(_ sz: Int, padding: Padding = .valid) {
        poolSize = (1, sz, sz, 1)
        strides = (1, sz, sz, 1)
        self.padding = padding
    }

    @differentiable
    public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return avgPool2D(input, filterSize: poolSize, strides: strides, padding: padding)
    }
}

# +
//export

@frozen
public struct FAGlobalAvgPool2D<Scalar: TensorFlowFloatingPoint>: FALayer {
    // TF-603 workaround.
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>
    @noDerivative public var delegates: [(Output) -> ()] = []
    
    public init() {}

    @differentiable
    public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return input.mean(squeezingAxes: [1,2])
    }
}
# -

# ### Make Array conform to Differentiable

//export
extension Array: Layer where Element: Layer, Element.Input == Element.Output {
    // Note: remove the explicit typealiases after TF-603 is resolved.
    public typealias Input = Element.Input
    public typealias Output = Element.Output
    
    @differentiable(vjp: _vjpApplied)
    public func callAsFunction(_ input: Input) -> Output {
        var activation = input
        for layer in self {
            activation = layer(activation)
        }
        return activation
    }
    
    public func _vjpApplied(_ input: Input)
        -> (Output, (Output.TangentVector) -> (Array.TangentVector, Input.TangentVector))
    {
        var activation = input
        var pullbacks: [(Input.TangentVector) -> (Element.TangentVector, Input.TangentVector)] = []
        for layer in self {
            let (newActivation, newPullback) = layer.valueWithPullback(at: activation) { $0($1) }
            activation = newActivation
            pullbacks.append(newPullback)
        }
        func pullback(_ v: Input.TangentVector) -> (Array.TangentVector, Input.TangentVector) {
            var activationGradient = v
            var layerGradients: [Element.TangentVector] = []
            for pullback in pullbacks.reversed() {
                let (newLayerGradient, newActivationGradient) = pullback(activationGradient)
                activationGradient = newActivationGradient
                layerGradients.append(newLayerGradient)
            }
            return (Array.TangentVector(layerGradients.reversed()), activationGradient)
        }
        return (activation, pullback)
    }
}

# ### Simplify some names

# +
//export 
extension KeyPathIterable {
    public var keyPaths: [WritableKeyPath<Self, Tensor<Float>>] {
        return recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)
    }
}

extension Layer {
    public var variables: AllDifferentiableVariables {
        get { return allDifferentiableVariables }
        set { allDifferentiableVariables = newValue }
    }
}
# -

# ## power operator

# +
// export
public func ** (lhs: Int, rhs: Int) -> Int {
    return Int(pow(Double(lhs), Double(rhs)))
}

public func ** (lhs: Double, rhs: Double) -> Double {
    return pow(lhs, rhs)
}

public func **<T : BinaryFloatingPoint>(_ x: T, _ y: T) -> T {
    return T(pow(Double(x), Double(y)))
}

public func **<T>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
  where T : TensorFlowFloatingPoint { return pow(x, y)}

public func **<T>(_ x: T, _ y: Tensor<T>) -> Tensor<T>
  where T : TensorFlowFloatingPoint { return pow(x, y)}

public func **<T>(_ x: Tensor<T>, _ y: T) -> Tensor<T>
  where T : TensorFlowFloatingPoint { return pow(x, y)}
# -

# ### Compose

//export
public extension Differentiable {
    @differentiable
    func compose<L1: Layer, L2: Layer>(_ l1: L1, _ l2: L2) -> L2.Output
        where L1.Input == Self, L1.Output == L2.Input {
        return sequenced(through: l1, l2)
    }
    
    @differentiable
    func compose<L1: Layer, L2: Layer, L3: Layer>(_ l1: L1, _ l2: L2, _ l3: L3) -> L3.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input {
        return sequenced(through: l1, l2, l3)
    }
    
    @differentiable
    func compose<L1: Layer, L2: Layer, L3: Layer, L4: Layer>(
        _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4
    ) -> L4.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input,
              L3.Output == L4.Input {
        return sequenced(through: l1, l2, l3, l4)
    }
    
    @differentiable
    func compose<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer>(
        _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5
    ) -> L5.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
              L4.Output == L5.Input {
        return sequenced(through: l1, l2, l3, l4, l5)
    }
    
    @differentiable
    func compose<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer>(
        _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6
    ) -> L6.Output
        where L1.Input == Self, L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
              L4.Output == L5.Input, L5.Output == L6.Input {
        return sequenced(through: l1, l2, l3, l4, l5, l6)
    }
}

# # Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"01a_fastai_layers.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


