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

# # Callbacks

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_03_minibatch_training")' FastaiNotebook_03_minibatch_training

//export
import Path
import TensorFlow

import FastaiNotebook_03_minibatch_training

# ## Load data

# We load our data and define a basic model like in the previous notebook.

var (xTrain,yTrain,xValid,yValid) = loadMNIST(path: mnistPath, flat: true)

let (n,m) = (xTrain.shape[0],xTrain.shape[1])
let c = yTrain.max().scalarized()+1
print(n,m,c)
let nHid = 50

// export
public struct BasicModel: Layer {
    public var layer1, layer2: FADense<Float>
    
    public init(nIn: Int, nHid: Int, nOut: Int){
        layer1 = FADense(nIn, nHid, activation: relu)
        layer2 = FADense(nHid, nOut)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return layer2(layer1(input))
    }
}

# We can also directly define our model as an array of `FADense` layers:

var model: [FADense<Float>] = [
    FADense(m, nHid, activation: relu),
    FADense(nHid, Int(c))] // BasicModel(nIn: m, nHid: nHid, nOut: Int(c))

# ### Dataset/DataBunch

# We add our own wrapper above the S4TF `Dataset` for several reasons:
# - in S4TF, `Dataset` has no length and we need a `count` property to be able to do efficient hyper-parameters scheduling.
# - you can only apply `batched` once to a `Dataset` but we sometimes want to change the batch size. We save the original non-batched datasetin `innerDs`.
# - the shuffle needs to be called each time we want to reshuffle, so we make this happen in the compute property `ds`.

//export 
public struct FADataset<Element> where Element: TensorGroup {
    public var innerDs: Dataset<Element>
    public var shuffle = false
    public var bs = 64 
    public var dsCount: Int
    
    public var count: Int {
        return dsCount%bs == 0 ? dsCount/bs : dsCount/bs+1
    }
    
    public var ds: Dataset<Element> { 
        if !shuffle { return innerDs.batched(bs)}
        let seed = Int64.random(in: Int64.min..<Int64.max)
        return innerDs.shuffled(sampleCount: dsCount, randomSeed: seed).batched(bs)
    }
    
    public init(_ ds: Dataset<Element>, len: Int, shuffle: Bool = false, bs: Int = 64) {
        (self.innerDs,self.dsCount,self.shuffle,self.bs) = (ds, len, shuffle, bs)
    }
}

# Then we can define a `DataBunch` to group our training and validation datasets.

// export
public struct DataBunch<Element> where Element: TensorGroup{
    public var train, valid: FADataset<Element>
    
    public init(train: Dataset<Element>, valid: Dataset<Element>, trainLen: Int, validLen: Int, bs: Int = 64) {
        self.train = FADataset(train, len: trainLen, shuffle: true,  bs: bs)
        self.valid = FADataset(valid, len: validLen, shuffle: false, bs: 2*bs)
    }
}

# And add a convenience function to get MNIST in a `DataBunch` directly.

//export
public func mnistDataBunch(path: Path = mnistPath, flat: Bool = false, bs: Int = 64)
   -> DataBunch<DataBatch<TF, TI>> {
    let (xTrain,yTrain,xValid,yValid) = loadMNIST(path: path, flat: flat)
    return DataBunch(train: Dataset(elements: DataBatch(xb:xTrain, yb: yTrain)), 
                     valid: Dataset(elements: DataBatch(xb:xValid, yb: yValid)),
                     trainLen: xTrain.shape[0],
                     validLen: xValid.shape[0],
                     bs: bs)
}

let data = mnistDataBunch(flat: true)

data.train.count

# ## Shuffle test

# Timing

//export
public extension Sequence {
  func first() -> Element? {
    return first(where: {_ in true})
  }
}

time(repeating: 10) {
  let tst = data.train.ds

  tst.first()!.yb
}

# Check we get different batches:

var tst = data.train.ds
tst.first()!.yb

tst = data.train.ds
tst.first()!.yb

# # `Learner`,  `LearnerAction`: enums and error handling in Swift, oh my!

# Just like in Python, we'll use "exception handling" to let custom actions indicate that they want to stop, skip over a batch or do other custom processing - e.g. for early stopping.
#
# We'll start by defining a custom type to represent the stop reason, and we'll use a Swift enum to describe it:

// export
public enum LearnerAction: Error {
    case skipEpoch(reason: String)
    case skipBatch(reason: String)
    case stop(reason: String)
}

# Now this a bit of an unusual thing - we have met protocols before, and `: Error` is a protocol that `LearnerAction` conforms to, but what is going on with those cases?
#
# Let's jump briefly into slides to talk about Swift enums:
#
# **Slides:** [Supercharged Enums in Swift](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g512a2e238a_144_147)
#
#

# ### Basic `Learner` class

// export
/// Initializes and trains a model on a given dataset.
public final class Learner<Label: TensorGroup,
                           Opt: TensorFlow.Optimizer & AnyObject>
    where Opt.Scalar: Differentiable,
          Opt.Model: Layer,
          // Constrain model input to Tensor<Float>, to work around
          // https://forums.fast.ai/t/fix-ad-crash-in-learner/42970.
          Opt.Model.Input == Tensor<Float>
{
    public typealias Model = Opt.Model
    public typealias Input = Model.Input
    public typealias Output = Model.Output
    public typealias Data = DataBunch<DataBatch<Input, Label>>
    public typealias Loss = TF
    public typealias Optimizer = Opt
    public typealias Variables = Model.AllDifferentiableVariables
    public typealias EventHandler = (Learner) throws -> Void
    
    /// A wrapper class to hold the loss function, to work around
    // https://forums.fast.ai/t/fix-ad-crash-in-learner/42970.
    public final class LossFunction {
        public typealias F = @differentiable (Model.Output, @nondiff Label) -> Loss
        public var f: F
        init(_ f: @escaping F) { self.f = f }
    }
    
    public var data: Data
    public var opt: Optimizer
    public var lossFunc: LossFunction
    public var model: Model
    
    public var currentInput: Input!
    public var currentTarget: Label!
    public var currentOutput: Output!
    
    public private(set) var epochCount = 0
    public private(set) var currentEpoch = 0
    public private(set) var currentGradient = Model.TangentVector.zero
    public private(set) var currentLoss = Loss.zero
    public private(set) var inTrain = false
    public private(set) var pctEpochs = Float.zero
    public private(set) var currentIter = 0
    public private(set) var iterCount = 0
    
    open class Delegate {
        open var order: Int { return 0 }
        public init () {}
        
        open func trainingWillStart(learner: Learner) throws {}
        open func trainingDidFinish(learner: Learner) throws {}
        open func epochWillStart(learner: Learner) throws {}
        open func epochDidFinish(learner: Learner) throws {}
        open func validationWillStart(learner: Learner) throws {}
        open func batchWillStart(learner: Learner) throws {}
        open func batchDidFinish(learner: Learner) throws {}
        open func didProduceNewGradient(learner: Learner) throws {}
        open func optimizerDidUpdate(learner: Learner) throws {}
        open func batchSkipped(learner: Learner, reason:String) throws {}
        open func epochSkipped(learner: Learner, reason:String) throws {}
        open func trainingStopped(learner: Learner, reason:String) throws {}
        ///
        /// TODO: learnerDidProduceNewOutput and learnerDidProduceNewLoss need to
        /// be differentiable once we can have the loss function inside the Learner
    }
    
    public var delegates: [Delegate] = [] {
        didSet { delegates.sort { $0.order < $1.order } }
    }
    
    public init(data: Data, lossFunc: @escaping LossFunction.F,
                optFunc: (Model) -> Optimizer, modelInit: ()->Model) {
        (self.data,self.lossFunc) = (data,LossFunction(lossFunc))
        model = modelInit()
        opt = optFunc(self.model)
    }
}

# Then let's write the parts of the training loop:

// export
extension Learner {
    private func evaluate(onBatch batch: DataBatch<Input, Label>) throws {
        currentOutput = model(currentInput)
        currentLoss = lossFunc.f(currentOutput, currentTarget)
    }
    
    private func train(onBatch batch: DataBatch<Input, Label>) throws {
        let (xb,yb) = (currentInput!,currentTarget!) //We still have to force-unwrap those for AD...
        (currentLoss, currentGradient) = model.valueWithGradient { model -> Loss in 
            let y = model(xb)                                      
            self.currentOutput = y
            return self.lossFunc.f(y, yb)
        }
        for d in delegates { try d.didProduceNewGradient(learner: self) }
        opt.update(&model.variables, along: self.currentGradient)
    }
    
    private func train(onDataset ds: FADataset<DataBatch<Input, Label>>) throws {
        iterCount = ds.count
        for batch in ds.ds {
            (currentInput, currentTarget) = (batch.xb, batch.yb)
            do {
                for d in delegates { try d.batchWillStart(learner: self) }
                if inTrain { try train(onBatch: batch) } else { try evaluate(onBatch: batch) }
            }
            catch LearnerAction.skipBatch(let reason) {
                for d in delegates {try d.batchSkipped(learner: self, reason:reason)}
            }
            for d in delegates { try d.batchDidFinish(learner: self) }
        }
    }
}

# And the whole fit function.

// export
extension Learner {
    /// Starts fitting.
    /// - Parameter epochCount: The number of epochs that will be run.
    public func fit(_ epochCount: Int) throws {
        self.epochCount = epochCount
        do {
            for d in delegates { try d.trainingWillStart(learner: self) }
            for i in 0..<epochCount {
                self.currentEpoch = i
                do {
                    for d in delegates { try d.epochWillStart(learner: self) }
                    try train(onDataset: data.train)
                    for d in delegates { try d.validationWillStart(learner: self) }
                    try train(onDataset: data.valid)
                } catch LearnerAction.skipEpoch(let reason) {
                    for d in delegates {try d.epochSkipped(learner: self, reason:reason)}
                }
                for d in delegates { try d.epochDidFinish(learner: self) }
            }
        } catch LearnerAction.stop(let reason) {
            for d in delegates {try d.trainingStopped(learner: self, reason:reason)}
        }

        for d in delegates { try d.trainingDidFinish(learner: self) }
    }
}

# ### Test

func optFunc(_ model: BasicModel) ->  SGD<BasicModel> { return SGD(for: model, learningRate: 1e-2)}

func modelInit() -> BasicModel {return BasicModel(nIn: m, nHid: nHid, nOut: Int(c))}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

learner.fit(2)

# ## Let's add Callbacks!

# Extension with convenience methods to add delegates:

// export
public extension Learner {
    func addDelegate (_ delegate :  Learner.Delegate ) { delegates.append(delegate) }
    func addDelegates(_ delegates: [Learner.Delegate]) { self.delegates += delegates }
}

# ### Train/eval

# Callback classes are defined as extensions of the Learner.

// export
extension Learner {
    public class TrainEvalDelegate: Delegate {
        public override func trainingWillStart(learner: Learner) {
            learner.pctEpochs = 0.0
        }

        public override func epochWillStart(learner: Learner) {
            Context.local.learningPhase = .training
            (learner.pctEpochs,learner.inTrain,learner.currentIter) = (Float(learner.currentEpoch),true,0)
        }
        
        public override func batchDidFinish(learner: Learner) {
            learner.currentIter += 1
            if learner.inTrain{ learner.pctEpochs += 1.0 / Float(learner.iterCount) }
        }
        
        public override func validationWillStart(learner: Learner) {
            Context.local.learningPhase = .inference
            learner.inTrain = false
            learner.currentIter = 0
        }
    }
    
    public func makeTrainEvalDelegate() -> TrainEvalDelegate { return TrainEvalDelegate() }
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

learner.delegates = [learner.makeTrainEvalDelegate()]

learner.fit(2)

# ### AverageMetric

// export
extension Learner {
    public class AvgMetric: Delegate {
        public let metrics: [(Output, Label) -> TF]
        var total: Int = 0
        var partials = [TF]()
        
        public init(metrics: [(Output, Label) -> TF]) { self.metrics = metrics}
        
        public override func epochWillStart(learner: Learner) {
            total = 0
            partials = Array(repeating: Tensor(0), count: metrics.count + 1)
        }
        
        public override func batchDidFinish(learner: Learner) {
            if !learner.inTrain{
                let bs = learner.currentInput!.shape[0] //Possible because Input is TF for now
                total += bs
                partials[0] += Float(bs) * learner.currentLoss
                for i in 1...metrics.count{
                    partials[i] += Float(bs) * metrics[i-1](learner.currentOutput!, learner.currentTarget!)
                }
            }
        }
        
        public override func epochDidFinish(learner: Learner) {
            for i in 0...metrics.count {partials[i] = partials[i] / Float(total)}
            print("Epoch \(learner.currentEpoch): \(partials)")
        }
    }
    
    public func makeAvgMetric(metrics: [(Output, Label) -> TF]) -> AvgMetric{
        return AvgMetric(metrics: metrics)
    }
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeAvgMetric(metrics: [accuracy])]

learner.fit(2)

# ### Normalization

// export
extension Learner {
    public class Normalize: Delegate {
        public let mean, std: TF
        public init(mean: TF, std: TF) { (self.mean,self.std) = (mean,std) }
        
        public override func batchWillStart(learner: Learner) {
            learner.currentInput = (learner.currentInput! - mean) / std
        }
    }
    
    public func makeNormalize(mean: TF, std: TF) -> Normalize{
        return Normalize(mean: mean, std: std)
    }
}

(mean: xTrain.mean(), std: xTrain.standardDeviation())

// export
public let mnistStats = (mean: TF(0.13066047), std: TF(0.3081079))

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeAvgMetric(metrics: [accuracy]),
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std)]

learner.fit(2)

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"04_callbacks.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


