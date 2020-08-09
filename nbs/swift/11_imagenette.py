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
%install '.package(path: "$cwd/FastaiNotebook_10_mixup_ls")' FastaiNotebook_10_mixup_ls

// export
import Path
import TensorFlow

import FastaiNotebook_10_mixup_ls

%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

# ## Load data

let path = downloadImagenette()

let il = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"])

# Then we split them according to the grandparent folder. `train` for the training set (which is the default) and `val` for the validation set.

let sd = SplitData(il) {grandParentSplitter(fName: $0, valid: "val")}

# We define our processors for the training set and the validation set. The difference with python is that we have to specify a noop processor (with a type) when we don't want to do anything.

var procLabel = CategoryProcessor()

# Then we can label our data using the parent directory and those two processors.

let sld = makeLabeledData(sd, fromFunc: parentLabeler, procLabel: &procLabel)

# We can then convert to a databunch by specifying two function that will convert our items and our labels to `Tensor`. For the items we use `pathsToTensor` that converts the `Path` object to their string representation then `StringTensor`. For the labels,  the function `intsToTensor` just convert the indices we have in proper tensors.

let rawData = sld.toDataBunch(itemToTensor: pathsToTensor, labelToTensor: intsToTensor, bs: 128)

# The main difference with python is that the transforms are all applied directly on the datasets by `tf.data`. Even opening on the image is such a transform, that will take a `StringTensor` and return a tensor of `UInt8`. We have written a function that opens the image in a filename and returns it decoded and resized to `size`, which is the transform we apply to our items.

let data = transformData(rawData) { openAndResize(fname: $0, size: 128) }

# We can then have a look by grabbing one batch.

let batch = data.train.oneBatch()!

# Our batches have two attributes `xb` and `yb` that contain the inputs and targets respectively.

print(batch.xb.shape)
print(batch.yb.shape)

# If we decode the labels suing our processor, we can plot images with their corresponding classes:

let labels = batch.yb.scalars.map { procLabel.vocab![Int($0)] }
showImages(batch.xb, labels: labels)

# ## XResnet

# We build the same xresnet we had in fastai over PyTorch. Modules in S4TF are `struct` that conform to the `Layer` protocol. You define any layer it uses as attributes that you have to properly set in the `init` function. The equivalent of `forward` in PyTorch is `callAsFunction`.
#
# We are using our custom fastai layers that have the prefix `FA` because they contain experimental features that might eventually be merged in S4TF. There is a NoBiasConv layer separate from the Conv Layer because S4TF doesn't yet support control flow. That means you can't have if statements or for loops in the `callAsFunction` function.

//export
public struct ConvLayer: Layer {
    public var bn: FABatchNorm<Float>
    public var conv: FANoBiasConv2D<Float>
    
    public init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 1, zeroBn: Bool = false, act: Bool = true){
        bn = FABatchNorm(featureCount: cOut)
//      "activation: act ? relu : identity" fails on 0.3.1, so we use if/else
        if act {conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu)}
        else   {conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: identity)}
        if zeroBn { bn.scale = Tensor(zeros: [cOut]) }
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        // TODO: Work around https://bugs.swift.org/browse/TF-606
        return bn.forward(conv.forward(input))
    }
}

# However we will need some optional layers to have refactored resnet implementation: in the shortcuts, we sometimes apply an average pool or a convolution layer, but most of the time we don't do anything. To be able to have that in S4TF, we write a customized protocol called `SwitchableLayer`. It inherits from `Layer` and adds a boolean `isOn` and a differentiable `forward` function. A structure conforming to it will return the result of `forward` if `isOn` is `true`, otherwise it won't do anything.
#
# As we said before, Swift autodiff doesn't support control flow (yet), so `if` statements aren't differentiable. That doesn't stop us from doing interesting work though, because we can create a custom function that will manually compute the gradients of our switchable layer:

# +
//export
//A layer that you can switch off to do the identity instead
public protocol SwitchableLayer: Layer {
    associatedtype Input
    var isOn: Bool { get set }
    
    @differentiable func forward(_ input: Input) -> Input
}

public extension SwitchableLayer {
    func callAsFunction(_ input: Input) -> Input {
        return isOn ? forward(input) : input
    }

    @differentiating(callAsFunction)
    func gradForward(_ input: Input) ->
           (value: Input,
            pullback: (Self.Input.TangentVector) ->
                                  (Self.TangentVector, Self.Input.TangentVector)) {
        if isOn {
            return valueWithPullback(at: input) { $0.forward($1) } 
        } else {
            return (input, { (Self.TangentVector.zero, $0) }) 
        }
    }
}
# -

# Then we use this protocol to create a `MaybeAvgPool2D` layer and a `MaybeConv` layer. The little downside is that we will have to create a fake convolution for the `MaybeConv` that are just identity, but we will just create a 1x1x1x1 weight matrix so that it doesn't take much memory.

//export
public struct MaybeAvgPool2D: SwitchableLayer {
    var pool: FAAvgPool2D<Float>
    @noDerivative public var isOn: Bool
    
    @differentiable public func forward(_ input: TF) -> TF { return pool(input) }
    
    public init(_ sz: Int) {
        isOn = (sz > 1)
        pool = FAAvgPool2D<Float>(sz)
    }
}

//export
public struct MaybeConv: SwitchableLayer {
    var conv: ConvLayer
    @noDerivative public var isOn: Bool
    
    @differentiable public func forward(_ input: TF) -> TF { return conv(input) }
    
    public init(_ cIn: Int, _ cOut: Int) {
        isOn = (cIn > 1) || (cOut > 1)
        conv = ConvLayer(cIn, cOut, ks: 1, act: false)
    }
}

# With those two maybe layers, we can write a `ResBlock` as we are used to. We can have array of layers that have the same type, and such an array can be treated as if it was a normal layer.

# +
//export
public struct ResBlock: Layer {
    public var convs: [ConvLayer]
    public var idConv: MaybeConv
    public var pool: MaybeAvgPool2D
    
    public init(_ expansion: Int, _ ni: Int, _ nh: Int, stride: Int = 1){
        let (nf, nin) = (nh*expansion,ni*expansion)
        convs = (expansion==1) ? [
            ConvLayer(nin, nh, ks: 3, stride: stride),
            ConvLayer(nh, nf, ks: 3, zeroBn: true, act: false)
        ] : [
            ConvLayer(nin, nh, ks: 1),
            ConvLayer(nh, nh, ks: 3, stride: stride),
            ConvLayer(nh, nf, ks: 1, zeroBn: true, act: false)
        ]
        idConv = nin==nf ? MaybeConv(1,1) : MaybeConv(nin, nf)
        pool = MaybeAvgPool2D(stride)
    }
    
    @differentiable
    public func callAsFunction(_ inp: TF) -> TF {
        return relu(convs(inp) + idConv(pool(inp)))
    }
    
}
# -

# Then we can create our `XResnet` pretty much the same way as in pytorch. The list comprehesions are replaced by uses of `map` or `reduce`. The `makeLayer` function is out side of the `XResNet` structure because it can't be used in the `init` otherwise (in swift, you can only use self in the init when all the attributes have been properly set, and this `makeLayer` function is used to create the attribute `blocks`).

//export
func makeLayer(_ expansion: Int, _ ni: Int, _ nf: Int, _ nBlocks: Int, stride: Int) -> [ResBlock] {
    return Array(0..<nBlocks).map { ResBlock(expansion, $0==0 ? ni : nf, nf, stride: $0==0 ? stride : 1) }
}

//export
public struct XResNet: Layer {
    public var stem: [ConvLayer]
    public var maxPool = MaxPool2D<Float>(poolSize: (3,3), strides: (2,2), padding: .same)
    public var blocks: [ResBlock]
    public var pool = GlobalAvgPool2D<Float>()
    public var linear: Dense<Float>
    
    public init(_ expansion: Int, _ layers: [Int], cIn: Int = 3, cOut: Int = 1000){
        var nfs = [cIn, (cIn+1)*8, 64, 64]
        stem = (0..<3).map{ ConvLayer(nfs[$0], nfs[$0+1], stride: $0==0 ? 2 : 1)}
        nfs = [64/expansion,64,128,256,512]
        blocks = layers.enumerated().map { (i,l) in 
            return makeLayer(expansion, nfs[i], nfs[i+1], l, stride: i==0 ? 1 : 2)
        }.reduce([], +)
        linear = Dense(inputSize: nfs.last!*expansion, outputSize: cOut)
    }
    
    @differentiable
    public func callAsFunction(_ inp: TF) -> TF {
        return inp.compose(stem, maxPool, blocks, pool, linear)
    }
}

//export
public func xresnet18 (cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(1, [2, 2, 2, 2], cIn: cIn, cOut: cOut) }
public func xresnet34 (cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(1, [3, 4, 6, 3], cIn: cIn, cOut: cOut) }
public func xresnet50 (cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(4, [3, 4, 6, 3], cIn: cIn, cOut: cOut) }
public func xresnet101(cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(4, [3, 4, 23, 3], cIn: cIn, cOut: cOut) }
public func xresnet152(cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(4, [3, 8, 36, 3], cIn: cIn, cOut: cOut) }

# To define a `Learner` we need our data, a model initializer function, and optimizer initializer function and a loss function. The model initilializer is a simple closure that returns the model.

func modelInit() -> XResNet { return xresnet18(cOut: 10) }

# The optimizer function is a convenience function we wrote in notebook 09 (like we had done in the python version) that returns a `StatefulOptimizer` with all the necessary stats/step delegates.

let optFunc: (XResNet) -> StatefulOptimizer<XResNet> = adamOpt(lr: 1e-3, mom: 0.9, beta: 0.99, wd: 1e-2, eps: 1e-6)

# Then we can create our `Learner`. The loss function is the classic cross entropy + softmax, and is given by S4TF.

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

# In swift, callbacks are called delegates. We have written a convenience function to automatically add the basic ones we need (train/eval, the recorder, metrics progress bar). This function returns the `recorder` if we want to look at losses or do some plots later.
#
# Then we add the delegate to normalize our inputs with the statistics of ImageNet.

let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.addDelegate(learner.makeNormalize(mean: imagenetStats.mean, std: imagenetStats.std))

# Then we can fit with the 1cycle policy:

learner.addOneCycleDelegates(1e-3, pctStart: 0.5)
learner.fit(5)

learner.addOneCycleDelegates(1e-3, pctStart: 0.5)
learner.fit(1)

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"11_imagenette.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


