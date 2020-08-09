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

# # Early stopping

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_05_anneal")' FastaiNotebook_05_anneal

//export
import Path
import TensorFlow
import Python

import FastaiNotebook_05_anneal

%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

# ## Load data

let data = mnistDataBunch(flat: true)

let (n,m) = (60000,784)
let c = 10
let nHid = 50

func optFunc(_ model: BasicModel) -> SGD<BasicModel> {return SGD(for: model, learningRate: 1e-2)}

func modelInit() -> BasicModel {return BasicModel(nIn: m, nHid: nHid, nOut: c)}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeRecorder()

# Check the previous callbacks load.

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeShowProgress(),
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std),
                     learner.makeAvgMetric(metrics: [accuracy]), recorder]

learner.fit(2)

# Make an extension to quickly load them. 

// export
//TODO: when recorder can be accessed as a property, remove it from the return
extension Learner where Opt.Scalar: PythonConvertible {
    public func makeDefaultDelegates(metrics: [(Output, Label) -> TF] = []) -> Recorder {
        let recorder = makeRecorder()
        delegates = [makeTrainEvalDelegate(), makeShowProgress(), recorder]
        if !metrics.isEmpty { delegates.append(makeAvgMetric(metrics: metrics)) }
        return recorder
    }
}

# ## Control Flow test

extension Learner {
    public class TestControlFlow: Delegate {
        public override var order: Int { return 3 }
        
        var skipAfter,stopAfter: Int
        public init(skipAfter:Int, stopAfter: Int){  (self.skipAfter,self.stopAfter) = (skipAfter,stopAfter) }
        
        public override func batchWillStart(learner: Learner) throws {
            print("batchWillStart")
            if learner.currentIter >= stopAfter {
                throw LearnerAction.stop(reason: "*** stopped: \(learner.currentIter)")
            }
            if learner.currentIter >= skipAfter {
                throw LearnerAction.skipBatch(reason: "*** skipBatch: \(learner.currentIter)")
            }
        }
        
        public override func trainingDidFinish(learner: Learner) {
            print("trainingDidFinish")
        }
        
        public override func batchSkipped(learner: Learner, reason: String) {
            print(reason)
        }
    }
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

learner.delegates = [type(of: learner).TestControlFlow(skipAfter:5, stopAfter: 8),
                     learner.makeTrainEvalDelegate()]

learner.fit(5)

# Check if the orders were taken into account:

(learner.delegates[0].order,learner.delegates[1].order)

# ### LR Finder

// export
extension Learner where Opt.Scalar: BinaryFloatingPoint {
    public class LRFinder: Delegate {
        public typealias ScheduleFunc = (Float) -> Float

        // A learning rate schedule from step to float.
        private var scheduler: ScheduleFunc
        private var numIter: Int
        private var minLoss: Float? = nil
        
        public init(start: Float = 1e-5, end: Float = 10, numIter: Int = 100) {
            scheduler = makeAnnealer(start: start, end: end, schedule: expSchedule)
            self.numIter = numIter
        }
        
        override public func batchWillStart(learner: Learner) {
            learner.opt.learningRate = Opt.Scalar(scheduler(Float(learner.currentIter)/Float(numIter)))
        }
        
        override public func batchDidFinish(learner: Learner) throws {
            if minLoss == nil {minLoss = learner.currentLoss.scalar}
            else { 
                if learner.currentLoss.scalarized() < minLoss! { minLoss = learner.currentLoss.scalarized()}
                if learner.currentLoss.scalarized() > 4 * minLoss! { 
                    throw LearnerAction.stop(reason: "Loss diverged")
                }
                if learner.currentIter >= numIter { 
                    throw LearnerAction.stop(reason: "Finished the range.") 
                }
            }
        }
        
        override public func validationWillStart(learner: Learner<Label, Opt>) throws {
            //Skip validation during the LR range test
            throw LearnerAction.skipEpoch(reason: "No validation in the LR Finder.")
        }
    }
    
    public func makeLRFinder(start: Float = 1e-5, end: Float = 10, numIter: Int = 100) -> LRFinder {
        return LRFinder(start: start, end: end, numIter: numIter)
    }
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates()

learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))
learner.delegates.append(learner.makeLRFinder())

learner.fit(2)

recorder.plotLRFinder()

// export
//TODO: when Recorder is a property of Learner don't return it.
extension Learner where Opt.Scalar: PythonConvertible & BinaryFloatingPoint {
    public func lrFind(start: Float = 1e-5, end: Float = 10, numIter: Int = 100) -> Recorder {
        let epochCount = data.train.count/numIter + 1
        let recorder = makeDefaultDelegates()
        delegates.append(makeLRFinder(start: start, end: end, numIter: numIter))
        try! self.fit(epochCount)
        return recorder
    }
}

let recorder = learner.lrFind()

recorder.plotLRFinder()

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"05b_early_stopping.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


