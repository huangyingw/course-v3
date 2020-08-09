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

%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_09_optimizer")' FastaiNotebook_09_optimizer

// export
import Path
import TensorFlow

import FastaiNotebook_09_optimizer

%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

# ## Load data

# //TODO: switch to imagenette when possible to train

let data = mnistDataBunch(flat: true)

let (n,m) = (60000,784)
let c = 10
let nHid = 50

func modelInit() -> BasicModel {return BasicModel(nIn: m, nHid: nHid, nOut: c)}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: sgdOpt(lr: 0.1), modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

learner.fit(1)

# ### Mixup

//export
extension RandomDistribution {
    // Returns a batch of samples.
    func next<G: RandomNumberGenerator>(
        _ count: Int, using generator: inout G
    ) -> [Sample] {
        var result: [Sample] = []
        for _ in 0..<count {
            result.append(next(using: &generator))
        }
        return result
    }

    // Returns a batch of samples, using the global Threefry RNG.
    func next(_ count: Int) -> [Sample] {
        return next(count, using: &ThreefryRandomNumberGenerator.global)
    }
}

# Mixup requires one-hot encoded targets since we don't have a loss function with no reduction.

//export
extension Learner {
    public class MixupDelegate: Delegate {
        private var distribution: BetaDistribution
        
        public init(alpha: Float = 0.4){
            distribution = BetaDistribution(alpha: alpha, beta: alpha)
        }
        
        override public func batchWillStart(learner: Learner) {
            if let xb = learner.currentInput {
                if let yb = learner.currentTarget as? Tensor<Float>{
                    var lambda = Tensor<Float>(distribution.next(Int(yb.shape[0])))
                    lambda = max(lambda, 1-lambda)
                    let shuffle = Raw.randomShuffle(value: Tensor<Int32>(0..<Int32(yb.shape[0])))
                    let xba = Raw.gather(params: xb, indices: shuffle)
                    let yba = Raw.gather(params: yb, indices: shuffle)
                    lambda = lambda.expandingShape(at: 1)
                    learner.currentInput = lambda * xb + (1-lambda) * xba
                    learner.currentTarget = (lambda * yb + (1-lambda) * yba) as? Label
                }
            }
        }
    }
    
    public func makeMixupDelegate(alpha: Float = 0.4) -> MixupDelegate {
        return MixupDelegate(alpha: alpha)
    }
}

let (n,m) = (60000,784)
let c = 10
let nHid = 50

# We need to one-hot encode the targets

var train1 = data.train.innerDs.map { DataBatch<TF,TF>(xb: $0.xb, 
                            yb: Raw.oneHot(indices: $0.yb, depth: TI(10), onValue: TF(1), offValue: TF(0))) }

var valid1 = data.valid.innerDs.map { DataBatch<TF,TF>(xb: $0.xb, 
                            yb: Raw.oneHot(indices: $0.yb, depth: TI(10), onValue: TF(1), offValue: TF(0))) }

let data1 = DataBunch(train: train1, valid: valid1, trainLen: data.train.dsCount, 
                  validLen: data.valid.dsCount, bs: data.train.bs)

func modelInit() -> BasicModel {return BasicModel(nIn: m, nHid: nHid, nOut: c)}

func accuracyFloat(_ out: TF, _ targ: TF) -> TF {
    return TF(out.argmax(squeezingAxis: 1) .== targ.argmax(squeezingAxis: 1)).mean()
}

let learner = Learner(data: data1, lossFunc: softmaxCrossEntropy, optFunc: sgdOpt(lr: 0.1), modelInit: modelInit)
let recorder = learner.makeRecorder()

learner.delegates = [learner.makeTrainEvalDelegate(), learner.makeShowProgress(), 
                     learner.makeAvgMetric(metrics: [accuracyFloat]), recorder,
                     learner.makeMixupDelegate(alpha: 0.2)]

learner.fit(2)

# ### Labelsmoothing

//export
@differentiable(wrt: out)
public func labelSmoothingCrossEntropy(_ out: TF, _ targ: TI, ε: Float = 0.1) -> TF {
    let c = out.shape[1]
    let loss = softmaxCrossEntropy(logits: out, labels: targ)
    let logPreds = logSoftmax(out)
    return (1-ε) * loss - (ε / Float(c)) * logPreds.mean()
}

@differentiable(wrt: out)
func lossFunc(_ out: TF, _ targ: TI) -> TF { return labelSmoothingCrossEntropy(out, targ, ε: 0.1) }

let learner = Learner(data: data, lossFunc: lossFunc, optFunc: sgdOpt(lr: 0.1), modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

learner.fit(2)

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"10_mixup_ls.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


