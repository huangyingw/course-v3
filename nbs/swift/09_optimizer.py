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
%install '.package(path: "$cwd/FastaiNotebook_08a_heterogeneous_dictionary")' FastaiNotebook_08a_heterogeneous_dictionary

// export
import Path
import TensorFlow

import FastaiNotebook_08a_heterogeneous_dictionary

%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

# ## Load data

let path = downloadImagenette()

let il = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"])
let sd = SplitData(il, fromFunc: {grandParentSplitter(fName: $0, valid: "val")})
var procLabel = CategoryProcessor()
let sld = makeLabeledData(sd, fromFunc: parentLabeler, procLabel: &procLabel)
let rawData = sld.toDataBunch(itemToTensor: pathsToTensor, labelToTensor: intsToTensor)
let data = transformData(rawData, tfmItem: { openAndResize(fname: $0, size: 128) })

func modelInit() -> CNNModel { return CNNModel(channelIn: 3, nOut: 10, filters: [64, 64, 128, 256]) }

# ## Stateful optimizer

# Before we begin, we create this structure to contain the names of our hyper-parameters. This will give us some tab completion and typo-proof way of handling them.

//export
public struct HyperParams {
    public static let lr = "learningRate"
}

# Like in the python version, we create `statDelegates` that will be responsible for computing/updating statistics in the state (like the moving average of gradients) and `stepDelegates` that will be responsible for performing a part of the update of the weights. 
#
# In PyTorch we created a basic class with functions that needed to be implemented. In swift this is what protocols are for.

# +
//export
public protocol StatDelegate {
    var name: String {get}
    var defaultHPs: [String:Float] {get}
    
    func update(_ state: inout [String:TF], p: TF, 𝛁p: TF, hps: inout [String:Float])
}

public protocol StepDelegate {
    var defaultHPs: [String:Float] {get}
    
    func update(_ p: inout TF, 𝛁p: inout TF, state: [String:TF], hps: inout [String:Float])
}
# -

# Those are helper functions to merge dictionaries that we'll use in the `StatefulOptimizer`.

# +
//export
public func mergeDicts(_ dicts: inout [[String:Float]], with newDict: [String:Float]) {
    for i in dicts.indices { 
        dicts[i].merge(newDict) { (_, new) in new } 
    }
}

public func mergeDicts(_ dicts: inout [[String:Float]], with newDicts: [[String:Float]]) {
    for i in dicts.indices { 
        dicts[i].merge(newDicts[i]) { (_, new) in new } 
    }
}
# -

# Those two extensions are there to initialize dicts easily.

# +
//export
extension Dictionary where Value == Int{
    public init(mapFromArrays arrays: [[Key]]){
        self.init(uniqueKeysWithValues: arrays.enumerated().flatMap { i, arr in arr.map { ($0, i) } })
    }
}

extension Dictionary {
    public init(constant: Value, keys: [Key]){
        self.init(uniqueKeysWithValues: keys.map { ($0, constant) })
    }
}
# -

# This is the initial state of our StatefulOptimizer. It's a dictionary keyPath (see below) to dictionary that maps names to tensor of floats.

//export
public func initState<Model: Layer>(for model: Model, names: [String]) 
-> [WritableKeyPath<Model.AllDifferentiableVariables, TF>: [String:TF]] {
    return [WritableKeyPath<Model.AllDifferentiableVariables, TF>: [String:TF]](
        constant: [String: TF](constant: TF(0), keys: names),
        keys: model.variables.keyPaths)
}

# And we can define the main `StatefulOptimizer`. It takes a model, hyperparameters for each parameter group, some `steppers` and `stats` and a `splitArray` that defines our different parameter groups.
#
# To understand how this work, you need to know a little bit about keyPaths. This is a tool in swift to access any elements in a nested structure like our models: one model typically has a few attributes that are modules which in turn contain other modules and so forth until we reach the primitives layers like `Conv2d` or `Dense`. KeyPaths will allow us to index in that nested structure the objects of a particular type. 
#
# For instance, the shortcut `keyPaths` you can apply to any `Layer` will find all the tensors of floats. If we apply it to a `Model.AllDifferentiableVariables` object, we will find all the parameters of the model (since `model.allDifferentiableVariables` only contain the trainable parameters).
#
# That's why the inner loop of our `StatefulOptimizer` is over `variables.keyPaths`. The same keyPath index will give us the gradients. Then we create a `state` to be a dictionary of such keyPaths to `[String:TF]` and the `splitArray` we provide is an array of different keyPaths (each giving us a parameter group) from which we build a `splitDict` that maps our keyPaths to the index of the corresponding group.

//export
public class StatefulOptimizer<Model: Layer>
    where Model.AllDifferentiableVariables == Model.TangentVector {
    public typealias ModelKeyPath = WritableKeyPath<Model.AllDifferentiableVariables, TF>
    public typealias SplitDict = [ModelKeyPath: Int]
    public var hpGroups: [[String:Float]]
    public var splitDict: SplitDict
    public var states: [ModelKeyPath: [String: TF]]
    public var stats: [StatDelegate]
    public var steppers: [StepDelegate]
    public init(        
        for model: __shared Model,
        steppers: [StepDelegate],
        stats: [StatDelegate],
        hpGroups: [[String:Float]],
        splitArray: [[ModelKeyPath]]
    ) {
        self.hpGroups = Array(repeating: [:], count: hpGroups.count)
        (self.steppers,self.stats) = (steppers,stats)
        self.splitDict = SplitDict(mapFromArrays: splitArray)
        states = [:]
        steppers.forEach { mergeDicts(&self.hpGroups, with: $0.defaultHPs) }
        stats.forEach    { mergeDicts(&self.hpGroups, with: $0.defaultHPs) }
        states = initState(for: model, names: stats.map { $0.name })
        mergeDicts(&self.hpGroups, with: hpGroups)
    }
        
    public func update(
        _ variables: inout Model.AllDifferentiableVariables,
        along direction: Model.TangentVector
    ) {
        for kp in variables.keyPaths {
            var 𝛁p = direction[keyPath: kp]
            var hps = hpGroups[splitDict[kp]!]
            stats.forEach() { $0.update(&states[kp]!, p: variables[keyPath: kp], 𝛁p: 𝛁p, hps: &hps) }
            steppers.forEach() { $0.update(&variables[keyPath: kp], 𝛁p: &𝛁p, state: states[kp]!, hps: &hps) }
            hpGroups[splitDict[kp]!] = hps
        }
    }
}

# To make `StatefulOptimizer` conform to the `Optimizer` protocol, we need to add a `learningRate` property.

//export
extension StatefulOptimizer: Optimizer{
    public var learningRate: Float {
        get { return hpGroups.last![HyperParams.lr]! } 
        set { 
            for i in hpGroups.indices {self.hpGroups[i][HyperParams.lr] = newValue }
        }
    }
    //For discriminative learning rates
    public var learningRates: [Float] {
        get { return hpGroups.map { $0[HyperParams.lr]! } }
        set { 
            for i in hpGroups.indices {self.hpGroups[i][HyperParams.lr] = newValue[i] } 
        }
    }
}

# When we don't have any parameter groups, we just use one with all the `keyPaths`. This convenience init automatically does that for us.

//export
extension StatefulOptimizer{
    public convenience init (for model: __shared Model,
                             steppers: [StepDelegate],
                             stats: [StatDelegate],
                             hps: [String:Float]) {
        self.init(for: model,
                  steppers: steppers,
                  stats: stats,
                  hpGroups: [hps],
                  splitArray: [model.variables.keyPaths])
    }
}

# We are now ready to define `steppers` and `stats`. Let's begin with basic SGD:

//export
public struct SGDStep: StepDelegate {
    public var defaultHPs: [String: Float] { return [HyperParams.lr: 3e-3] }
    public init() {}
    public func update(_ p: inout TF, 𝛁p: inout TF, state: [String:TF], hps: inout [String:Float]) {
        p -= 𝛁p * hps[HyperParams.lr]!
    }
}

# We can check all is working and train:

var hps: [String:Float] = [HyperParams.lr: 0.01]
func optFunc(_ model: CNNModel) -> StatefulOptimizer<CNNModel> {
    return StatefulOptimizer(for: model, steppers: [SGDStep()], stats: [], hps: hps)
}

var learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
var recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

learner.fit(1)

# Then we can add weight decay and L2 regularization.

# +
//export
public extension HyperParams {
    static let wd = "weightDecay"
}

public struct WeightDecay: StepDelegate {
    public var defaultHPs: [String: Float] { return [HyperParams.wd: 0] }
    public init() {}
    public func update(_ p: inout TF, 𝛁p: inout TF, state: [String:TF], hps: inout [String:Float]) {
        p *= 1 - hps[HyperParams.lr]! * hps[HyperParams.wd]!
    }
}
# -

//export
public struct L2Regularization: StepDelegate {
    public var defaultHPs: [String: Float] { return [HyperParams.wd: 0] }
    public init() {}
    public func update(_ p: inout TF, 𝛁p: inout TF, state: [String:TF], hps: inout [String:Float]) {
        𝛁p += hps[HyperParams.wd]! * p
    }
}

# The next step is SGD with momentum. For this we need a statistic that keeps track of the moving average of the gradients.

//export
//Expandable enum to have tab completes/typo-proof for state variable names.
public struct StateKeys {
    public static let avgGrad = "averageGrad"
}

# +
//export
public extension HyperParams {
    static let mom = "momentum"
    static let momDamp = "dampening"
}

public struct AverageGrad: StatDelegate {
    public var defaultHPs: [String: Float] { return [HyperParams.mom: 0.9] }
    public let dampened: Bool
    public init(dampened: Bool = false) { self.dampened = dampened }
    public var name: String { return StateKeys.avgGrad }
    public func update(_ state: inout [String: TF], p: TF, 𝛁p: TF, hps: inout [String:Float]) {
        state[StateKeys.avgGrad]! *= hps[HyperParams.mom]!
        hps[HyperParams.momDamp] = 1.0 - (dampened ? hps[HyperParams.mom]! : 0.0)
        state[StateKeys.avgGrad]! += hps[HyperParams.momDamp]! * 𝛁p
    }
}
# -

//export
public struct MomentumStep: StepDelegate {
    public var defaultHPs: [String: Float] = [:]
    public init() {}
    public func update(_ p: inout TF, 𝛁p: inout TF, state: [String: TF], hps: inout [String:Float]) {
        p -= state[StateKeys.avgGrad]! * hps[HyperParams.lr]!
    }
}

# And we can check it trains properly.

let hps: [String:Float] = [HyperParams.lr: 0.01]
func optFunc(_ model: CNNModel) -> StatefulOptimizer<CNNModel> {
    return StatefulOptimizer(for: model, steppers: [MomentumStep()], stats: [AverageGrad()], hps: hps)
}

var learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
var recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

learner.fit(1)

# The hyper-parameters have taken the default values provided (except for learning rates).

learner.opt.hpGroups[0]

# The next step is Adam. For that we need to keep track of the averages of the gradients squared.

# +
//export
public extension HyperParams {
    static let ²mom = "momentumSquares"
    static let ²momDamp = "dampeningSquares"
}

public extension StateKeys {
    static let avgSqr = "averageSquaredGrad"
}

public struct AverageSquaredGrad: StatDelegate {
    let dampened: Bool
    public init(dampened: Bool = true) { self.dampened = dampened }
    public var name: String { return StateKeys.avgSqr }
    public var defaultHPs: [String: Float] { return [HyperParams.²mom: 0.99] }
    public func update(_ state: inout [String: TF], p: TF, 𝛁p: TF, hps: inout [String:Float]) {
        state[StateKeys.avgSqr]! *= hps[HyperParams.²mom]!
        hps[HyperParams.²momDamp] = 1.0 - (dampened ? hps[HyperParams.²mom]! : 0.0)
        state[StateKeys.avgSqr]! += hps[HyperParams.²momDamp]! * 𝛁p.squared()
    }
}
# -

# And we also need to keep track of the number of iterations we did.

# +
//export
public extension StateKeys {
    static let step = "stepCount"
}

public struct StepCount: StatDelegate {
    public var name: String { return StateKeys.step }
    public var defaultHPs: [String:Float] = [:]
    public init() {}
    public func update(_ state: inout [String: TF], p: TF, 𝛁p: TF, hps: inout [String:Float]) {
        state[StateKeys.step]! += 1.0
    }
}
# -

//export
//public struct Epsilon: HetDictKey { public static var defaultValue: Float = 1e-5 }
public extension HyperParams {
    static let eps = "epsilon"
}

//export
public struct AdamStep: StepDelegate {
    public var defaultHPs: [String: Float] { return [HyperParams.eps: 1e-5] }
    public init() {}
    public func update(_ p: inout TF, 𝛁p: inout TF, state: [String: TF], hps: inout [String:Float]) {
        let step = state[StateKeys.step]!
        let (mom,damp) = (hps[HyperParams.mom]!,hps[HyperParams.momDamp]!)
        let debias1 = damp * (1 - pow(mom, step)) / (1 - mom)
        let num = debias1 * state[StateKeys.avgGrad]!
        
        let (²mom,²damp) = (hps[HyperParams.²mom]!,hps[HyperParams.²momDamp]!)
        let debias2 = ²damp * (1 - pow(²mom, step)) / (1 - ²mom)
        let denom = sqrt(state[StateKeys.avgSqr]!/debias2) + hps[HyperParams.eps]!
        
        p -= hps[HyperParams.lr]! * num / denom
    }
}

# Again let's check it's all training properly.

func optFunc(_ model: CNNModel) -> StatefulOptimizer<CNNModel> {
    return StatefulOptimizer(
        for: model,
        steppers: [AdamStep()], 
        stats: [AverageGrad(dampened: true), AverageSquaredGrad(), StepCount()], 
        hps: [HyperParams.lr: 1e-3])
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

learner.fit(1)

# We can also check the values of the hyper-parameters have been set properly.

learner.opt.hpGroups[0]

# Defining the Lamb optimizer is as easy as before.

public struct LambStep: StepDelegate {
    public var defaultHPs: [String: Float] { return [HyperParams.eps: 1e-6, HyperParams.wd: 0.0] }
    public func update(_ p: inout TF, 𝛁p: inout TF, state: [String: TF], hps: inout [String:Float]) {
        let stepCount = state[StateKeys.step]!
        let (mom,damp) = (hps[HyperParams.mom]!,hps[HyperParams.momDamp]!)
        let debias1 = damp * (1 - pow(mom, stepCount)) / (1 - mom)
        let num = debias1 * state[StateKeys.avgGrad]!
        
        let (²mom,²damp) = (hps[HyperParams.²mom]!,hps[HyperParams.²momDamp]!)
        let debias2 = ²damp * (1 - pow(²mom, stepCount)) / (1 - ²mom)
        let denom = sqrt(state[StateKeys.avgSqr]!/debias2) + hps[HyperParams.eps]!
        
        let step = num / denom + hps[HyperParams.wd]! * p
        let r1 = sqrt((p * p).mean())
        let r2 = sqrt((step * step).mean())
        let factor = min(r1 / r2, Float(10.0))
        p -= hps[HyperParams.lr]! * factor * step
    }
}

# ### Making convenience functions

# To easily create our optimizers, we have two convenience functions.

// export
public func sgdOpt<Model>(lr: Float, mom: Float = 0.9, wd: Float = 0.0, dampening: Bool = false
                         ) -> ((Model) -> StatefulOptimizer<Model>) {
    var steppers: [StepDelegate] = (mom != 0) ? [MomentumStep()] : [SGDStep()]
    if wd != 0 { steppers.append(WeightDecay()) }
    let stats = (mom != 0) ? [AverageGrad(dampened: dampening)] : []
    var hps: [String: Float] = [HyperParams.lr: lr]
    if mom != 0 { hps[HyperParams.mom] = mom }
    if wd != 0  { hps[HyperParams.wd ] = wd  }
    return {model in 
        return StatefulOptimizer(for: model, steppers: steppers, stats: stats, hps: hps)}
}

// export
public func adamOpt<Model>(lr: Float, mom: Float = 0.9, beta: Float=0.99, wd: Float = 0.0, eps: Float = 1e-5
                         ) -> ((Model) -> StatefulOptimizer<Model>) {
    var steppers: [StepDelegate] = [AdamStep()]
    if wd != 0 { steppers.append(WeightDecay()) }
    let stats: [StatDelegate] = [AverageGrad(dampened: true), AverageSquaredGrad(), StepCount()]
    var hps: [String: Float] = [HyperParams.lr: lr]
    hps[HyperParams.mom] = mom
    hps[HyperParams.²mom] = beta
    hps[HyperParams.eps] = eps
    if wd != 0  { hps[HyperParams.wd ] = wd  }
    return {model in 
        return StatefulOptimizer(for: model, steppers: steppers, stats: stats, hps: hps)}
}

# ### Schedule the hyperparams

# The next thing is that we need to schedule our hyper-parameters. The following function allows us to schedule any of them, as long as they are present in the `hpGroups` dictionaries.

// export
public extension StatefulOptimizer {
    func setParam(_ hp: String, _ val: Float) {
        for i in 0..<hpGroups.count { hpGroups[i][hp] = val }
    }
}

// export
extension Learner where Opt.Scalar: BinaryFloatingPoint, 
    Opt.Model.AllDifferentiableVariables == Opt.Model.TangentVector{
    public class ParamScheduler: Delegate {
        public override var order: Int { return 1 }
        public typealias ScheduleFunc = (Float) -> Float

        // A learning rate schedule from step to float.
        public var scheduler: ScheduleFunc
        public let hp: String
        
        public init(scheduler: @escaping (Float) -> Float, hp: String) {
            (self.scheduler,self.hp) = (scheduler,hp)
        }
        
        override public func batchWillStart(learner: Learner) {
            let val = scheduler(learner.pctEpochs/Float(learner.epochCount))
            (learner.opt as! StatefulOptimizer<Opt.Model>).setParam(hp, val)
        }
    }
    
    public func makeParamScheduler(_ scheduler: @escaping (Float) -> Float, hp: String) -> ParamScheduler {
        return ParamScheduler(scheduler: scheduler, hp: hp)
    }
}

# We can then define a helper function to schedule a 1cycle policy.

// export 
public func oneCycleSchedulers(_ lrMax: Float, pctStart:Float=0.25, divStart: Float = 10, divEnd: Float = 1e5, 
                               moms: (Float,Float,Float) = (0.95,0.85,0.95)) 
-> ((Float) -> Float, (Float) -> Float){
    let lrSched = combineSchedules(
        pcts: [pctStart, 1-pctStart], 
        schedules: [makeAnnealer(start: lrMax/divStart, end: lrMax, schedule: cosineSchedule),
                    makeAnnealer(start: lrMax, end: lrMax/divEnd, schedule: cosineSchedule)])
    let momSched = combineSchedules(
        pcts: [pctStart, 1-pctStart], 
        schedules: [makeAnnealer(start: moms.0, end: moms.1, schedule: cosineSchedule),
                    makeAnnealer(start: moms.1, end: moms.2, schedule: cosineSchedule)])
    return (lrSched, momSched)
}

// export
extension Learner where Opt.Scalar: BinaryFloatingPoint, 
    Opt.Model.AllDifferentiableVariables == Opt.Model.TangentVector{

    public func addOneCycleDelegates(_ lrMax: Float, pctStart:Float=0.25, divStart: Float = 10, divEnd: Float = 1e5, 
                               moms: (Float,Float,Float) = (0.95,0.85,0.95)) {
        let scheds = oneCycleSchedulers(lrMax, pctStart: pctStart, divStart: divStart, divEnd: divEnd, moms: moms)
        addDelegates([makeParamScheduler(scheds.0 , hp: HyperParams.lr), 
                      makeParamScheduler(scheds.1 , hp: HyperParams.mom)])
    }
}

# And check it's all training properly.

let optFunc: (CNNModel) -> StatefulOptimizer<CNNModel> = adamOpt(lr: 1e-3, mom: 0.9, beta: 0.99, wd: 1e-2, eps: 1e-6)

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

learner.addOneCycleDelegates(1e-3)
learner.fit(1)

recorder.plotLRs()

# ### Differential learning rates

# To train at differential learning rates (or freeze part of the models) we need to pass to our optimizer arrays of KeyPaths (which will define our layer groups). For instance, we can begin with the firt 9 keyPaths (which corresponds to the first three ConvBNs):

func modelInit() -> CNNModel { return CNNModel(channelIn: 3, nOut: 10, filters: [64, 64, 128, 256]) }

var model = modelInit()
let splitArray = [Array(model.variables.keyPaths[0..<9]), Array(model.variables.keyPaths[9...])]

let hpGroups: [[String: Float]] = [[HyperParams.lr: 0], [HyperParams.lr: 0.1]]
func optFunc(_ model: CNNModel) -> StatefulOptimizer<CNNModel> {
    return StatefulOptimizer(for: model, steppers: [SGDStep()], stats: [], hpGroups: hpGroups, splitArray: splitArray)
}

let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))

# First parameter shouldn't change since the corresponding layer group as a LR of 0., second should.

learner.model.convs[0].norm.scale

learner.model.convs[3].norm.scale

learner.fit(1)

learner.model.convs[0].norm.scale

learner.model.convs[3].norm.scale

# Another way to get those keyPaths is to use the keyPaths to certain layers then append the keyPaths of all the parameters inside. This function takes a model, a layer and the keyPath that points from `model.variables` to `layer.variables` and returns the keyPaths of all the parameters of that layer. 

public func parameterKeyPaths<M1, M2>(
    _ model: M1,
    _ kp: WritableKeyPath<M1.AllDifferentiableVariables, M2.AllDifferentiableVariables>,
    _ layer: M2) -> [WritableKeyPath<M1.AllDifferentiableVariables, TF>]
where M1: Layer, M2: Layer {
    return model.variables[keyPath: kp].keyPaths.map { kp.appending(path: $0) }
}

# To access a keyPath directly, we use \ commands. Here is the keyPath to the array of convs, which lets us easily get the keyPaths for all the body of our CNN:

let kp = \(CNNModel.AllDifferentiableVariables).convs
let conv = model.convs
let bodyKeyPaths = parameterKeyPaths(model, kp, conv)

# Then we could split body and head:

let splitArray = [bodyKeyPaths, model.variables.keyPaths.filter { return !bodyKeyPaths.contains($0) }]
splitArray.map { $0.count }

# If we want to refine this a bit and split our body between the first 4 convs and the last 3 we can proceed like this:

let x = [1,2,3]
let y = [4,5,6]
zip(x,y).map { print($0, $1) }

# +
let deepBody = (0..<4).map { parameterKeyPaths(
    model, 
    \(CNNModel.AllDifferentiableVariables).convs.base[$0], 
    model.convs[$0]
) }.reduce([], +)

let upperBody = (4..<7).map { parameterKeyPaths(
    model, 
    \(CNNModel.AllDifferentiableVariables).convs.base[$0], 
    model.convs[$0]
) }.reduce([], +)
# -

let splitArray = [deepBody, upperBody, model.variables.keyPaths.filter { return !bodyKeyPaths.contains($0) }]
splitArray.map { $0.count }

# Now let's say we want a parameter group will all the batchnorm layers. KeyPaths allow us to get all the batchnorm layers by saying `(to: FABatchNorm<Float>.self)`. The `.keyPaths` method we have been using is jsut a shortcut for `recursivelyAllWritableKeyPaths(to: TF.self)`, which grabs all the keypaths to all the tensors.

let bns = model.recursivelyAllWritableKeyPaths(to: FABatchNorm<Float>.self).map { model[keyPath: $0] }

# Then we need the keypaths from model.variables to those batchnorms.

let bnKeyPaths = model.variables.recursivelyAllWritableKeyPaths(to: FABatchNorm<Float>.AllDifferentiableVariables.self)

let bnParameters = zip(bnKeyPaths, bns).map { parameterKeyPaths(model, $0, $1) }.reduce([], +)
bnParameters.count

# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"09_optimizer.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


