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

# # Minibatch training

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_02a_why_sqrt5")' FastaiNotebook_02a_why_sqrt5

//export
import Path
import TensorFlow

import FastaiNotebook_02a_why_sqrt5

# Our labels will be integeres from now on, so to go with our `TF` abbreviation, we introduce `TI`.

// export
public typealias TI = Tensor<Int32>

# ### Data

# We gather the MNIST data like in the previous notebooks.

var (xTrain,yTrain,xValid,yValid) = loadMNIST(path: Path.home/".fastai"/"data"/"mnist_tst", flat: true)

let trainMean = xTrain.mean()
let trainStd  = xTrain.std()

xTrain = normalize(xTrain, mean: trainMean, std: trainStd)
xValid = normalize(xValid, mean: trainMean, std: trainStd)

let (n,m) = (xTrain.shape[0],xTrain.shape[1])
let c = yTrain.max().scalarized()+1
print(n,m,c)

# We also define a simple model using our `FADense` layers.

let nHid = 50

public struct MyModel: Layer {
    public var layer1: FADense<Float>
    public var layer2: FADense<Float>
    
    public init(nIn: Int, nHid: Int, nOut: Int){
        layer1 = FADense(nIn, nHid, activation: relu)
        layer2 = FADense(nHid, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: TF) -> TF {
        return input.sequenced(through: layer1, layer2)
    }
}

var model = MyModel(nIn: m, nHid: nHid, nOut: Int(c))

let pred = model(xTrain)

# ### Cross entropy loss

# Before we can train our model, we need to have a loss function. We saw how to write `logSoftMax` from scratch in PyTorch, but let's do it once in swift too.

func logSoftmax<Scalar>(_ activations: Tensor<Scalar>) -> Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    let exped = exp(activations) 
    return log(exped / exped.sum(alongAxes: -1))
}

let smPred = logSoftmax(pred)

yTrain[0..<3]

(smPred[0][5],smPred[1][0],smPred[2][4])

# There is no fancy indexing yet so we have to use gather to get the indices we want out of our softmaxed predictions.

func nll<Scalar>(_ input: Tensor<Scalar>, _ target :TI) -> Tensor<Scalar> 
    where Scalar:TensorFlowFloatingPoint{
        let idx: TI = Raw.range(start: Tensor(0), limit: Tensor(numericCast(target.shape[0])), delta: Tensor(1))
        let indices = Raw.concat(concatDim: Tensor(1), [idx.expandingShape(at: 1), target.expandingShape(at: 1)])
        let losses = Raw.gatherNd(params: input, indices: indices)
        return -losses.mean()
    }

nll(smPred, yTrain)

time(repeating: 100){ let _ = nll(smPred, yTrain) }

# Simplify `logSoftmax` with log formulas.

func logSoftmax<Scalar>(_ activations: Tensor<Scalar>) -> Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    return activations - log(exp(activations).sum(alongAxes: -1))
}

let smPred = logSoftmax(pred)

nll(smPred, yTrain)

# We know use the LogSumExp trick

smPred.max(alongAxes: -1).shape

func logSumExp<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    let m = x.max(alongAxes: -1)
    return m + log(exp(x-m).sum(alongAxes: -1))
}

func logSoftmax<Scalar>(_ activations: Tensor<Scalar>) -> Tensor<Scalar> where Scalar:TensorFlowFloatingPoint{
    return activations - logSumExp(activations)
}

let smPred = logSoftmax(pred)

nll(smPred, yTrain)

# In S4TF nll loss is combined with softmax in:

let loss = softmaxCrossEntropy(logits: pred, labels: yTrain)
loss

time(repeating: 100){ _ = nll(logSoftmax(pred), yTrain)}

time(repeating: 100){ _ = softmaxCrossEntropy(logits: pred, labels: yTrain)}

# ## Basic training loop

# Basically the training loop repeats over the following steps:
# - get the output of the model on a batch of inputs
# - compare the output to the labels we have and compute a loss
# - calculate the gradients of the loss with respect to every parameter of the model
# - update said parameters with those gradients to make them a little bit better

// export
public func accuracy(_ output: TF, _ target: TI) -> TF{
    let corrects = TF(output.argmax(squeezingAxis: 1) .== target)
    return corrects.mean()
}

# We have a raw model for now, so it should be as good as random: 10% accuracy.

print(accuracy(pred, yTrain))

# So let's begin with a minibatch.

let bs=64                     // batch size
let xb = xTrain[0..<bs]       // a mini-batch from x
let preds = model(xb)         // predictions
print(preds[0], preds.shape)

# Then we can compute a loss

let yb = yTrain[0..<bs]
let loss = softmaxCrossEntropy(logits: preds, labels: yb)

print(accuracy(preds, yb))

let lr:Float = 0.5   // learning rate
let epochs = 1       // how many epochs to train for

# Then we can get our loss and gradients.

# Sometimes you'll see closures written this way (required if there is >1 statement in it).

let (loss, grads) = model.valueWithGradient { model -> TF in
    let preds = model(xb)
    return softmaxCrossEntropy(logits: preds, labels: yb)
}

# The full loop by hand would look like this:

for epoch in 1 ... epochs {
    for i in 0 ..< (n-1)/bs {
        let startIdx = i * bs
        let endIdx = startIdx + bs
        let xb = xTrain[startIdx..<endIdx]
        let yb = yTrain[startIdx..<endIdx]
        let (loss, grads) = model.valueWithGradient {
            softmaxCrossEntropy(logits: $0(xb), labels: yb)
        }
        model.layer1.weight -= lr * grads.layer1.weight
        model.layer1.bias   -= lr * grads.layer1.bias
        model.layer2.weight -= lr * grads.layer2.weight
        model.layer2.bias   -= lr * grads.layer2.bias
    }
}

let preds = model(xValid)
accuracy(preds, yValid)

# `>80%` in one epoch, not too bad!

# We use a shorcut: `model.variables` stands for `model.allDifferentiableVariables` in S4TF. It extracts from our model a new struct with only the trainable parameters. For instance if `model` is a BatchNorm layer, it has four tensor of floats: running mean, runing std, weights and bias. The corresponding `model.variables` only has the weights and bias tensors.
#
# When we get the gradients of our model, we have another structure of the same type, and it's possible to perform basic arithmetic on those structures to make the update step super simple:

for epoch in 1 ... epochs {
    for i in 0 ..< (n-1)/bs {
        let startIdx = i * bs
        let endIdx = startIdx + bs
        let xb = xTrain[startIdx..<endIdx]
        let yb = yTrain[startIdx..<endIdx]
        let (loss, grads) = model.valueWithGradient {
            softmaxCrossEntropy(logits: $0(xb), labels: yb)
        }
        model.variables -= grads.scaled(by: lr)
    }
}

# Then we can use a S4TF optimizer to do the step for us (which doesn't win much just yet - but will be nice when we can use momentum, adam, etc). An optimizer takes a `Model.AllDifferentiableVariables` object and some gradients, and will perform the update.

let optimizer = SGD(for: model, learningRate: lr)

# Here's a handy function (thanks for Alexis Gallagher) to grab a batch of indices at a time.

//export
public func batchedRanges(start:Int, end:Int, bs:Int) -> UnfoldSequence<Range<Int>,Int>
{
  return sequence(state: start) { (batchStart) -> Range<Int>? in
    let remaining = end - batchStart
    guard remaining > 0 else { return nil}
    let currentBs = min(bs,remaining)
    let batchEnd = batchStart.advanced(by: currentBs)
    defer {  batchStart = batchEnd  }
    return batchStart ..< batchEnd
  }
}

for epoch in 1 ... epochs{
    for b in batchedRanges(start: 0, end: n, bs: bs) {
        let (xb,yb) = (xTrain[b],yTrain[b])
        let (loss, grads) = model.valueWithGradient {
            softmaxCrossEntropy(logits: $0(xb), labels: yb)
        }
        optimizer.update(&model.variables, along: grads)
    }
}

# ## Dataset

# We can create a swift `Dataset` from our arrays. It will automatically batch things for us:

// export
public struct DataBatch<Inputs: Differentiable & TensorGroup, Labels: TensorGroup>: TensorGroup {
    public var xb: Inputs
    public var yb: Labels
    
    public init(xb: Inputs, yb: Labels){ (self.xb,self.yb) = (xb,yb) }
}

let trainDs = Dataset(elements:DataBatch(xb:xTrain, yb:yTrain)).batched(bs)

for epoch in 1...epochs{
    for batch in trainDs {
        let (loss, grads) = model.valueWithGradient {
            softmaxCrossEntropy(logits: $0(xb), labels: yb)
        }
        optimizer.update(&model.variables, along: grads)
    }
}

# This `Dataset` can also do the shuffle for us:

for epoch in 1...epochs{
    for batch in trainDs.shuffled(sampleCount: yTrain.shape[0], randomSeed: 42){
        let (loss, grads) = model.valueWithGradient {
            softmaxCrossEntropy(logits: $0(xb), labels: yb)
        }
        optimizer.update(&model.variables, along: grads)
    }
}

# ### Training loop

# With everything before, we can now write a generic training loop. It needs two generic types: the optimizer (`Opt`) and the labels (`Label`):

public func train<Opt: Optimizer, Label:TensorGroup>(
    _ model: inout Opt.Model,
    on ds: Dataset<DataBatch<Opt.Model.Input, Label>>,
    using opt: inout Opt,
    lossFunc: @escaping @differentiable (Opt.Model.Output, @nondiff Label) -> Tensor<Opt.Scalar>
) where Opt.Model: Layer,
        Opt.Model.Input: TensorGroup,
        Opt.Model.TangentVector == Opt.Model.AllDifferentiableVariables,
        Opt.Scalar: TensorFlowFloatingPoint
{
    for batch in ds {
        let (loss, 𝛁model) = model.valueWithGradient {
            lossFunc($0(batch.xb), batch.yb)
        }
        opt.update(&model.variables, along: 𝛁model)
    }
}

var model = MyModel(nIn: m, nHid: nHid, nOut: Int(c))
var optimizer = SGD(for: model, learningRate: lr)

train(&model, on: trainDs, using: &optimizer, lossFunc: softmaxCrossEntropy)

let preds = model(xValid)
accuracy(preds, yValid)

# ### Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"03_minibatch_training.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


