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

# # Annealing

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_04_callbacks")' FastaiNotebook_04_callbacks

//export
import Path
import TensorFlow

import FastaiNotebook_04_callbacks

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

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeAvgMetric(metrics: [accuracy]),
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std)]

//Crashes! See: SR-10436
//learner.delegates = [type(of: learner).TrainEvalDelegate(), type(of: learner).AvgMetric(metrics: [accuracy])]

learner.fit(2)

# ## Recoder

# The Recorder's role is to keep track of the loss and our scheduled learning rate. 

// export
import Python
public let np = Python.import("numpy")
public let plt = Python.import("matplotlib.pyplot")

// export
public func plot<S1, S2>(_ arr1: [S1], _ arr2: [S2], logScale:Bool = false, xLabel: String="", yLabel: String = "") 
    where S1:PythonConvertible, S2:PythonConvertible{
    plt.figure(figsize: [6,4])
    let (npArr1, npArr2) = (np.array(arr1), np.array(arr2))
    if logScale {plt.xscale("log")} 
    if !xLabel.isEmpty {plt.xlabel(xLabel)}
    if !yLabel.isEmpty {plt.ylabel(yLabel)}    
    let fig = plt.plot(npArr1, npArr2)
    plt.show(fig)
}

// export
extension Learner where Opt.Scalar: PythonConvertible{
    public class Recorder: Delegate {
        public var losses: [Loss] = []
        public var lrs: [Opt.Scalar] = []
        
        public override func batchDidFinish(learner: Learner) {
            if learner.inTrain {
                losses.append(learner.currentLoss)
                lrs.append(learner.opt.learningRate)
            }
        }
        
        public func plotLosses(){
            plot(Array(0..<losses.count), losses.map{$0.scalar}, xLabel:"iteration", yLabel:"loss")
        }
        
        public func plotLRs(){
            plot(Array(0..<lrs.count), lrs, xLabel:"iteration", yLabel:"lr")
        }
        
        public func plotLRFinder(){
            plot(lrs, losses.map{$0.scalar}, logScale: true, xLabel:"lr", yLabel:"loss")
        }
        
    }
    
    public func makeRecorder() -> Recorder {
        return Recorder()
    }
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

# Utility optional property to get backour `Recorder` if it was created by a utility function. This doesn't always work properly for unkwnon reasons

//TODO: Fix
extension Learner where Opt.Scalar: PythonConvertible{
    public var recorder: Learner.Recorder? {
        for callback in learner.delegates {
            if let recorder = callback as? Learner.Recorder { return recorder }
        }
        return nil
    }
}

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeAvgMetric(metrics: [accuracy]), 
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std), learner.makeRecorder()]

learner.fit(2)

learner.recorder!.plotLosses()

# ### Progress bar

# It's nice to keep track of where we're at in the training with a progress bar.

// export
import Foundation

// export
func formatTime(_ t: Float) -> String {
    let t = Int(t)
    let (h,m,s) = (t/3600, (t/60)%60, t%60)
    return h != 0 ? String(format: "%02d:%02d:%02d", h, m, s) : String(format: "%02d:%02d", m, s)
}

formatTime(78.23)

// export
public struct ProgressBar{
    let total: Int
    let length: Int = 50
    let showEvery: Float = 0.2
    let fillChar: Character = "X"
    public var comment: String = ""
    private var waitFor: Int = 0
    private var startTime: UInt64 = 0
    private var lastPrint: UInt64 = 0
    private var lastShow: UInt64 = 0
    private var estimatedTotal: Float = 0.0
    private var bar: String = ""
    
    public init(_ c: Int) { total = c }
    
    public mutating func update(_ val: Int){
        lastShow = DispatchTime.now().uptimeNanoseconds
        if val == 0 { startTime = lastShow } 
        else {
            let averageTime = Float(lastShow - startTime) / (1e9 * Float(val))
            estimatedTotal = Float(total) * averageTime
        }
        if val == 0 || lastShow - lastPrint >= Int(1e9 * showEvery) { update_bar(val) }
    }
    
    public mutating func update_bar(_ val: Int){
        lastPrint = lastShow
        let prevLength = bar.count
        bar = String(repeating: fillChar, count: (val * length) / total)
        bar += String(repeating: "-", count: length - (val * length) / total)
        let pct = String(format: "%.2f", 100.0 * Float(val)/Float(total))
        let elapsedTime = Float(lastShow - startTime) / 1e9
        let remaingTime = estimatedTotal - elapsedTime
        bar += " \(pct)% [\(val)/\(total) \(formatTime(elapsedTime))<\(formatTime(remaingTime))"
        bar += comment.isEmpty ? "]" : " \(comment)]"
        if bar.count < prevLength { bar += String(repeating: " ", count: prevLength-bar.count) }
        print(bar, terminator:"\r")
        fflush(stdout)
    }
    
    public func remove(){
        print(String(repeating: " ", count: bar.count), terminator:"\r")
        fflush(stdout)
    }
}

var tst = ProgressBar(100)
for i in 0...100{
    tst.update(i)
    usleep(50000)
}
tst.remove()

// export
extension Learner {
    public class ShowProgress: Delegate {
        var pbar: ProgressBar? = nil
        var iter: Int = 0
        
        public override func epochWillStart(learner: Learner) {
            pbar = ProgressBar(learner.data.train.count)
        }
        
        public override func validationWillStart(learner: Learner) {
            if pbar != nil { pbar!.remove() }
            pbar = ProgressBar(learner.data.valid.count)
        }
        
        public override func epochDidFinish(learner: Learner) {
            if pbar != nil { pbar!.remove() }
        }
        
        public override func batchWillStart(learner: Learner) {
            if learner.currentIter == 0 {pbar!.update(0)}
        }
        
        public override func batchDidFinish(learner: Learner) {
            pbar!.update(learner.currentIter)
        }
        
        public override func trainingDidFinish(learner: Learner) {
            if pbar != nil { pbar!.remove() }
        }
    }
    
    public func makeShowProgress() -> ShowProgress { return ShowProgress() }
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeShowProgress(), 
                     learner.makeAvgMetric(metrics: [accuracy]), learner.makeRecorder(),
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std)]

learner.fit(2)

# ### Annealing

// export
/// A non-generalized learning rate scheduler
extension Learner where Opt.Scalar: BinaryFloatingPoint {
    public class LRScheduler: Delegate {
        public override var order: Int { return 1 }
        public typealias ScheduleFunc = (Float) -> Float

        // A learning rate schedule from step to float.
        public var scheduler: ScheduleFunc
        
        public init(scheduler: @escaping (Float) -> Float) {
            self.scheduler = scheduler
        }
        
        override public func batchWillStart(learner: Learner) {
            learner.opt.learningRate = Opt.Scalar(scheduler(learner.pctEpochs/Float(learner.epochCount)))
        }
    }
    
    public func makeLRScheduler(scheduler: @escaping (Float) -> Float) -> LRScheduler {
        return LRScheduler(scheduler: scheduler)
    }
}

# +
// export
public func linearSchedule(start: Float, end: Float, pct: Float) -> Float {
    return start + pct * (end - start)
}

public func makeAnnealer(start: Float, end: Float, schedule: @escaping (Float, Float, Float) -> Float) -> (Float) -> Float { 
    return { pct in return schedule(start, end, pct) }
}
# -

let annealer = makeAnnealer(start: 1e-2, end: 0.1, schedule: linearSchedule)
annealer(0.3)

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeRecorder()

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeShowProgress(), 
                     learner.makeAvgMetric(metrics: [accuracy]), recorder,
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std),
                     learner.makeLRScheduler(scheduler: annealer)]

learner.fit(2)

recorder.plotLRs()

# More annealing functions

# +
// export
public func constantSchedule(start: Float, end: Float, pct: Float) -> Float {
    return start
}

public func cosineSchedule(start: Float, end: Float, pct: Float) -> Float {
    return start + (1 + cos(Float.pi*(1-pct))) * (end-start) / 2
}

public func expSchedule(start: Float, end: Float, pct: Float) -> Float {
    return start * pow(end / start, pct)
}
# -

// export
public func combineSchedules(pcts: [Float], schedules: [(Float) -> Float]) -> ((Float) -> Float){
    var cumPcts: [Float] = [0]
    for pct in pcts {cumPcts.append(cumPcts.last! + pct)}
    func inner(pct: Float) -> Float{
        if (pct == 0.0) { return schedules[0](0.0) }
        if (pct > 1.0)  { return schedules.last!(1.0) }
        let i = cumPcts.firstIndex(where: {$0 >= pct})! - 1
        let actualPos = (pct-cumPcts[i]) / (cumPcts[i+1]-cumPcts[i])
        return schedules[i](actualPos)
    }
    return inner
}

let mySchedule = combineSchedules(pcts: [0.3, 0.7], 
                                  schedules: [makeAnnealer(start: 0.3, end: 0.6, schedule: cosineSchedule),
                                              makeAnnealer(start: 0.6, end: 0.2, schedule: cosineSchedule)])

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeRecorder()

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeShowProgress(), 
                     learner.makeAvgMetric(metrics: [accuracy]), recorder,
                     learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std),
                     learner.makeLRScheduler(scheduler: mySchedule)]

learner.fit(2)

recorder.plotLRs()

//Needs fixing 
//learner.recorder!.plotLRs()

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"05_anneal.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


