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

# # Data block foundations

# Uncomment line below when using Colab (this installs OpenCV4)
# %system SwiftCV/install/install_colab.sh
%install-location $cwd/swift-install
%install '.package(path: "$cwd/FastaiNotebook_07_batchnorm")' FastaiNotebook_07_batchnorm
%install '.package(path: "$cwd/SwiftCV")' SwiftCV

//export
import Path
import TensorFlow
import Python

import FastaiNotebook_07_batchnorm

%include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

# ## Image ItemList

# ### Download Imagenette

# First things first, we need to download Imagenette and untar it. What follows is very close to what we did for MNIST.

//export
public let dataPath = Path.home/".fastai"/"data"

//export
public func downloadImagenette(path: Path = dataPath) -> Path {
    let url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz"
    let fname = "imagenette-160"
    let file = path/fname
    try! path.mkdir(.p)
    if !file.exists {
        downloadFile(url, dest:(path/"\(fname).tgz").string)
        _ = "/bin/tar".shell("-xzf", (path/"\(fname).tgz").string, "-C", path.string)
    }
    return file
}

let path = downloadImagenette()

# If we look at `path.ls()`, we see it returns a list of entries, which are structures with a `kind` and a `path` attribute. The `kind` is an enum that can be `file` or `directory`. `path` then points to the corresponding location.

for e in path.ls() { print("\(e.path) (\(e.kind == .directory ? "directory": "file"))")}

for e in (path/"val").ls() { print("\(e.path) (\(e.kind == .directory ? "directory": "file"))")}

# Let's have a look inside a class folder (the first class is tench):

let pathTench = path/"val"/"n01440764"

let imgFn = Path.home/".fastai/data/imagenette-160/val/n01440764/ILSVRC2012_val_00006697.JPEG"
imgFn.string

# We will use opencv to read and resize our images.

//export
import SwiftCV
import Foundation

//load the image in memory
let imgContent = Data(contentsOf: imgFn.url)
// make opencv image
var cvImg = imdecode(imgContent)
// convert to RGB
cvImg = cvtColor(cvImg, nil, ColorConversionCode.COLOR_BGR2RGB)

# By converting this image to a tensor then numpy, we can use `plt` to plot it:

let tensImg = Tensor<UInt8>(cvMat: cvImg)!
let numpyImg = tensImg.makeNumpyArray()
plt.imshow(numpyImg) 
plt.axis("off")
plt.show()

# ### Grab all the images

# Now that we have donloaded the data, we need to be able to recursively grab all the filenames in the imagenette folder. The following function walks recursively through the folder and adds the filenames that have the right extension.

//export
public func fetchFiles(path: Path, recurse: Bool = false, extensions: [String]? = nil) -> [Path] {
    var res: [Path] = []
    for p in try! path.ls(){
        if p.kind == .directory && recurse { 
            res += fetchFiles(path: p.path, recurse: recurse, extensions: extensions)
        } else if extensions == nil || extensions!.contains(p.path.extension.lowercased()) {
            res.append(p.path)
        }
    }
    return res
}

# Note that we don't have a generic `open_image` function like in python here, but will be using a specific decode function (here for jpegs, but there is one for gifs or pngs). That's why we limit ourselves to jpeg exensions here.

time { let fNames = fetchFiles(path: path, recurse: true, extensions: ["jpeg", "jpg"]) }

let fNames = fetchFiles(path: path, recurse: true, extensions: ["jpeg", "jpg"])

fNames.count == 13394

print(type(of: cvImg))

# ## Prepare the data

//export
public protocol ItemBase {
    func transform( _ tfms: [(inout Self) -> ()]) -> Self
}

public struct Image: ItemBase{
    
    public var path: Path
    public lazy var img: Mat = {
        //print("Load image in memory")
        return imdecode(try! Data(contentsOf: path.url))
    } ()
    
    public init(_ img: Mat, _ path: Path) {
        (self.path,self.img) = (path,img)
    }
    
    public init(_ path: Path) { self.path = path }
    
    public mutating func show(){
        let tensImg = Tensor<UInt8>(cvMat: img)!
        let numpyImg = tensImg.makeNumpyArray()
        plt.imshow(numpyImg) 
        plt.axis("off")
        plt.show()
    }
    
    public func transform(_ tfms: [(inout Image) -> ()]) -> Image{
        var tfmedImg = Image(path)
        tfms.forEach() { $0(&tfmedImg) }
        return tfmedImg
    }
    
    public mutating func toTensor() -> TF {
        return TF(Tensor<UInt8>(cvMat: img)!)
    }
}

var img = Image(imgFn)

img.show()

# ### ItemList

// export
public struct ItemList<T> where T: ItemBase{
    public var items: [T]
    public let path: Path
    public var tfms: [(inout T) -> ()] = [] 
    
    public init(items: [T], path: Path, tfms: [(inout T) -> ()] = []){
        (self.items,self.path,self.tfms) = (items,path,tfms)
    }
    
    public init (_ il: ItemList<T>, newItems: [T]) {
        self.init(items: newItems, path: il.path, tfms: il.tfms)
    }
    
    public subscript(index: Int) -> T {
        return items[index].transform(tfms)
    }
}

// export
public protocol InitableFromPath {
    init(_ path: Path)
}
extension Image: InitableFromPath {}

// export
public extension ItemList where T: InitableFromPath {
    init(fromFolder path: Path, extensions: [String], recurse: Bool = true, tfms: [(inout T) -> ()] = []) {
        self.init(items: fetchFiles(path: path, recurse: recurse, extensions: extensions).map { T($0) },
                  path:  path,
                  tfms: tfms)
    }
}

let il: ItemList<Image> = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"])

var img = il[0]
img.show()

# +
func convertRGB(_ img: inout Image) {
    img.img = cvtColor(img.img, nil, ColorConversionCode.COLOR_BGR2RGB)
}

func resize(_ img: inout Image, size: Int) {
    img.img = resize(img.img, nil, Size(size, size), 0, 0, InterpolationFlag.INTER_AREA)
}
# -

let il: ItemList<Image> = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"],
                                  tfms: [convertRGB, { resize(&$0, size:128) }])

var img = il[0]
img.show()

# ### Split

// export
public struct SplitData<T> where T: ItemBase{
    public let train, valid: ItemList<T>
    public var path: Path { return train.path }
    
    public init(train: ItemList<T>, valid: ItemList<T>){
        (self.train, self.valid) = (train, valid)
    }
    
    public init(_ il: ItemList<T>, fromFunc: (T) -> Bool){
        self.init(train: ItemList(il, newItems: il.items.filter { !fromFunc($0) }),
                  valid: ItemList(il, newItems: il.items.filter {  fromFunc($0) }))
    }
}

// export
public func grandParentSplitter(fName: Path, valid: String = "valid") -> Bool{
    return fName.parent.parent.basename() == valid
}

let sd = SplitData(il) { grandParentSplitter(fName: $0.path, valid: "val") }

var img = sd.train[0]
img.show()

# ### Processor

// export
public protocol Processor {
    associatedtype Input: ItemBase
    associatedtype Output: ItemBase
    
    mutating func initState(_ items: [Input])
    func process1(_ item: Input) -> Output
    func deprocess1(_ item: Output) -> Input
}

// export
public extension Processor {
    func process(_ items: [Input]) -> [Output] {
        return items.map { process1($0) }
    }
    
    func deprocess(_ items: [Output]) -> [Input] {
        return items.map { deprocess1($0) }
    }
}

// export
public struct NoopProcessor<Item>: Processor where Item: ItemBase{
    public init() {}
   
    public mutating func initState(_ items: [Item]) {}
    
    public func process1  (_ item: Item) -> Item { return item }
    public func deprocess1(_ item: Item) -> Item { return item }
}

# +
//export
extension String: ItemBase {
    public func transform(_ tfms: [(inout String) -> ()]) -> (String) { return self }
}

extension Int: ItemBase {
    public func transform(_ tfms: [(inout Int) -> ()]) -> (Int) { return self }
}
# -

// export
public struct CategoryProcessor: Processor {
    public init() {}
    public var vocab: [String]? = nil
    public var reverseMap: [String: Int]? = nil
    
    public mutating func initState(_ items: [String]) {
        vocab = Array(Set(items)).sorted()
        reverseMap = [:]
        for (i,x) in vocab!.enumerated() { reverseMap![x] = i }
    }
    
    public func process1  (_ item: String) -> Int { return reverseMap![item]! }
    public func deprocess1(_ item: Int) -> String { return vocab![item] }
}

# ### Label

# When we build the datasets, we don't need to return a tupe (item, label) but to have the tensor(s) with the items and the tensor(s) with the labels separately.

# +
public struct LabeledItemList<I, L> where I:ItemBase, L: ItemBase{
    public var inputs: ItemList<I>
    public var labels: ItemList<L>
    public var path: Path { return inputs.path }
    
    public init(inputs: ItemList<I>, labels: ItemList<L>) {
        (self.inputs,self.labels) = (inputs,labels)
    }
    
    public subscript(_ i: Int) -> (I, L) {
        return (inputs[i], labels[i])
    }
}

public extension LabeledItemList {
    init(_ il: ItemList<I>, labelWithFunc f: @escaping (I) -> L) {
        self.init(inputs: il, labels: ItemList(items: il.items.map{ f($0) }, path: il.path, tfms: []))
    }
}
# -

public func parentLabeler(_ fName: Path) -> String { return fName.parent.basename() }

let ll = LabeledItemList(il, labelWithFunc: { parentLabeler($0.path) })

var x = ll[0]
x.0.show()
print(x.1)

public func process<PI> (_ il: ItemList<PI.Input>, proc: PI) -> ItemList<PI.Output> where PI: Processor {
    return ItemList(items: proc.process(il.items), path: il.path, tfms: [])
}

public func process<PI, PL> (_ lil: LabeledItemList<PI.Input, PL.Input>, procInp: PI, procLab: PL) 
-> LabeledItemList<PI.Output, PL.Output> where PI: Processor, PL: Processor {
    return LabeledItemList(
        inputs: process(lil.inputs, proc: procInp),
        labels: process(lil.labels, proc: procLab)
    )
}

public struct SplitLabeledData<PI, PL> where PI: Processor, PL: Processor{
    public var train, valid: LabeledItemList<PI.Output, PL.Output>
    public var path: Path { return train.path }
    public var procInp: PI
    public var procLab: PL
    
    public init(_ rawTrain: LabeledItemList<PI.Input,PL.Input>, 
                _ rawValid: LabeledItemList<PI.Input,PL.Input>,
                procInp: inout PI,
                procLab: inout PL) {
        procInp.initState(rawTrain.inputs.items)
        procLab.initState(rawTrain.labels.items)
        train = process(rawTrain, procInp: procInp, procLab: procLab)
        valid = process(rawValid, procInp: procInp, procLab: procLab)
        (self.procInp,self.procLab) = (procInp,procLab)
    }
}

public extension SplitLabeledData {
    init(_ sd: SplitData<PI.Input>, 
         labelWithFunc f: @escaping (PI.Input) -> PL.Input,
         procInp: inout PI,
         procLab: inout PL) {
        self.init(LabeledItemList(sd.train, labelWithFunc: f),
                  LabeledItemList(sd.valid, labelWithFunc: f),
                  procInp: &procInp,
                  procLab: &procLab)
    }
}

var procInp = NoopProcessor<Image>()
var procLab = CategoryProcessor()

var sld = SplitLabeledData(sd, labelWithFunc: { parentLabeler($0.path) }, procInp: &procInp, procLab: &procLab)

# Labeling loses the transforms.

var x = sld.train[0]
x.0.show()
print(sld.procLab.deprocess1(x.1))

# So we add them back

sld.train.inputs.tfms = [convertRGB, { resize(&$0, size:128) }]

var x = sld.train[0]
x.0.show()
print(sld.procLab.deprocess1(x.1))

public extension SplitLabeledData{
    mutating func transform(_ tfms: ([(inout PI.Output) -> ()], [(inout PI.Output) -> ()])){
        train.inputs.tfms = tfms.0
        valid.inputs.tfms = tfms.1
    }
}

var sld = SplitLabeledData(sd, labelWithFunc: { parentLabeler($0.path) }, procInp: &procInp, procLab: &procLab)
let tfms = [convertRGB, { resize(&$0, size:512) }]
sld.transform((tfms, tfms))

var x = sld.train[0]
x.0.show()
print(sld.procLab.deprocess1(x.1))

# What's below doesn't work with that's above.

func loadSync(_ n: Int) -> TF {
    var imgs: [TF] = []
    for i in 1...n { 
        var img = sld.train[i].0
        imgs.append(img.toTensor().expandingShape(at: 0) / 255.0)
    }
    return TF(concatenating: imgs, alongAxis: 0)
}

time { let imgs = loadSync(100) }

func loadQSync(_ n: Int) -> [Image] {
    var imgs: [Image] = []
    let queue = DispatchQueue(label: "myqueue")
    queue.sync {
        for i in 1...n { imgs.append(sld.train[i].0) }
    }
    return imgs
}

time { let imgs = loadQSync(100) }

func loadAsync(_ n: Int) -> [Image] {
    var imgs: [Image] = []
    let group = DispatchGroup()
    group.enter()
    let queue = DispatchQueue(label: "myqueue")
    queue.async {
        for i in 1...n { imgs.append(sld.train[i].0) }
        group.leave()
    }
    group.wait()
    return imgs
}

time { let imgs = loadAsync(100) }


