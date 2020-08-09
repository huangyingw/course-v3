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
%install '.package(url: "https://github.com/mxcl/Path.swift", from: "0.16.1")' Path
%install '.package(url: "https://github.com/saeta/Just", from: "0.7.2")' Just
%install '.package(url: "https://github.com/latenitesoft/NotebookExport", from: "0.5.0")' NotebookExport

# Currently there's a bug in swift-jupyter which requires we define any custom operators here:

# +
// export
precedencegroup ExponentiationPrecedence {
    associativity: right
    higherThan: MultiplicationPrecedence
}
infix operator ** : ExponentiationPrecedence

precedencegroup CompositionPrecedence { associativity: left }
infix operator >| : CompositionPrecedence
# -

# ## Getting the MNIST dataset

# We will use a couple of libraries here:
# - [Just](https://github.com/JustHTTP/Just) does the equivalent of requests in python
# - [Path](https://github.com/mxcl/Path.swift) is even better than its python counterpart
# - Foundation is the base library from Apple

//export
import Foundation
import Just
import Path

# We will need to gunzip, untar or unzip files we download, so instead of grabbing one library for each, we implement a function that can execute any shell command.

//export
public extension String {
    @discardableResult
    func shell(_ args: String...) -> String
    {
        let (task,pipe) = (Process(),Pipe())
        task.executableURL = URL(fileURLWithPath: self)
        (task.arguments,task.standardOutput) = (args,pipe)
        do    { try task.run() }
        catch { print("Unexpected error: \(error).") }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: String.Encoding.utf8) ?? ""
    }
}

print("/bin/ls".shell("-lh"))

# To download a file, we use the `Just` library.

//export
public func downloadFile(_ url: String, dest: String? = nil, force: Bool = false) {
    let dest_name = dest ?? (Path.cwd/url.split(separator: "/").last!).string
    let url_dest = URL(fileURLWithPath: (dest ?? (Path.cwd/url.split(separator: "/").last!).string))
    if !force && Path(dest_name)!.exists { return }

    print("Downloading \(url)...")

    if let cts = Just.get(url).content {
        do    {try cts.write(to: URL(fileURLWithPath:dest_name))}
        catch {print("Can't write to \(url_dest).\n\(error)")}
    } else {
        print("Can't reach \(url)")
    }
}

downloadFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz")

# Then we will need to read our data and convert it into a `Tensor`:

//export
import TensorFlow

# The following is generic over the element type on the return value.  We could define two functions like this:
#
# ```swift
# func loadMNIST(training: Bool, labels: Bool, path: Path, flat: Bool) -> Tensor<Float> {
# func loadMNIST(training: Bool, labels: Bool, path: Path, flat: Bool) -> Tensor<Int32> {
# ```
#
# but that would be boring.  So we make loadMNIST take a "type parameter" `T` which indicates what sort of element type to load a tensor into.

func loadMNIST<T>(training: Bool, labels: Bool, path: Path, flat: Bool) -> Tensor<T> {
    let split = training ? "train" : "t10k"
    let kind = labels ? "labels" : "images"
    let batch = training ? 60000 : 10000
    let shape: TensorShape = labels ? [batch] : (flat ? [batch, 784] : [batch, 28, 28])
    let dropK = labels ? 8 : 16
    let baseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    let fname = split + "-" + kind + "-idx\(labels ? 1 : 3)-ubyte"
    let file = path/fname
    if !file.exists {
        downloadFile("\(baseUrl)\(fname).gz", dest:(path/"\(fname).gz").string)
        "/bin/gunzip".shell("-fq", (path/"\(fname).gz").string)
    }
    let data = try! Data(contentsOf: URL(fileURLWithPath: file.string)).dropFirst(dropK)
    if labels { return Tensor(data.map(T.init)) }
    else      { return Tensor(data.map(T.init)).reshaped(to: shape)}
}

# But this doesn't work because S4TF can't just put any type of data inside a `Tensor`. We have to tell it that this type:
# - is a type that TF can understand and deal with
# - is a type that can be applied to the data we read in the byte format
#
# We do this by defining a protocol called `ConvertibleFromByte` that inherits from `TensorFlowScalar`. That takes care of the first requirement. The second requirement is dealt with by asking for an `init` method that takes `UInt8`:

//export
protocol ConvertibleFromByte: TensorFlowScalar {
    init(_ d:UInt8)
}

# Then we need to say that `Float` and `Int32` conform to that protocol. They already have the right initializer so we don't have to code anything.

//export
extension Float : ConvertibleFromByte {}
extension Int32 : ConvertibleFromByte {}

# Lastly, we write a convenience method for all types that conform to the `ConvertibleFromByte` protocol, that will convert some raw data to a `Tensor` of that type.

//export
extension Data {
    func asTensor<T:ConvertibleFromByte>() -> Tensor<T> {
        return Tensor(map(T.init))
    }
}

# And now we can write a generic `loadMNIST` function that can returns tensors of `Float` or `Int32`.

# +
//export
func loadMNIST<T: ConvertibleFromByte>
            (training: Bool, labels: Bool, path: Path, flat: Bool) -> Tensor<T> {
    let split = training ? "train" : "t10k"
    let kind = labels ? "labels" : "images"
    let batch = training ? 60000 : 10000
    let shape: TensorShape = labels ? [batch] : (flat ? [batch, 784] : [batch, 28, 28])
    let dropK = labels ? 8 : 16
    let baseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    let fname = split + "-" + kind + "-idx\(labels ? 1 : 3)-ubyte"
    let file = path/fname
    if !file.exists {
        downloadFile("\(baseUrl)\(fname).gz", dest:(path/"\(fname).gz").string)
        "/bin/gunzip".shell("-fq", (path/"\(fname).gz").string)
    }
    let data = try! Data(contentsOf: URL(fileURLWithPath: file.string)).dropFirst(dropK)
    if labels { return data.asTensor() }
    else      { return data.asTensor().reshaped(to: shape)}
}

public func loadMNIST(path:Path, flat:Bool = false)
        -> (Tensor<Float>, Tensor<Int32>, Tensor<Float>, Tensor<Int32>) {
    try! path.mkdir(.p)
    return (
        loadMNIST(training: true,  labels: false, path: path, flat: flat) / 255.0,
        loadMNIST(training: true,  labels: true,  path: path, flat: flat),
        loadMNIST(training: false, labels: false, path: path, flat: flat) / 255.0,
        loadMNIST(training: false, labels: true,  path: path, flat: flat)
    )
}
# -

# We will store mnist in this folder so that we don't download it each time we run a notebook:

//export
public let mnistPath = Path.home/".fastai"/"data"/"mnist_tst"

# The default returns mnist in the image format:

let (xTrain, yTrain, xValid, yValid) = loadMNIST(path: mnistPath)
xTrain.shape

# We can also ask for it in its flattened form:

let (xTrain, yTrain, xValid, yValid) = loadMNIST(path: mnistPath, flat: true)
xTrain.shape

# ## Timing

# Here is our time function:

# +
//export 
import Dispatch

// ⏰Time how long it takes to run the specified function, optionally taking
// the average across a number of repetitions.
public func time(repeating: Int = 1, _ f: () -> ()) {
    guard repeating > 0 else { return }
    
    // Warmup
    if repeating > 1 { f() }
    
    var times = [Double]()
    for _ in 1...repeating {
        let start = DispatchTime.now()
        f()
        let end = DispatchTime.now()
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
        let milliseconds = nanoseconds / 1e6
        times.append(milliseconds)
    }
    print("average: \(times.reduce(0.0, +)/Double(times.count)) ms,   " +
          "min: \(times.reduce(times[0], min)) ms,   " +
          "max: \(times.reduce(times[0], max)) ms")
}
# -

time(repeating: 10) {
    _ = loadMNIST(training: false, labels: false, path: mnistPath, flat: false) as Tensor<Float>
}

# ## Export

# Searching for a specific pattern with a regular expression isn't easy in swift. The good thing is that with an extension, we can make it easy for us!

// export
public extension String {
    func findFirst(pat: String) -> Range<String.Index>? {
        return range(of: pat, options: .regularExpression)
    }
    func hasMatch(pat: String) -> Bool {
        return findFirst(pat:pat) != nil
    }
}

# The foundation library isn't always the most convenient to use... This is how the first line of the following cell is written in it.
#
# ```swift
# let url_fname = URL(fileURLWithPath: fname)
# let last = fname.lastPathComponent
# let out_fname = (url_fname.deletingLastPathComponent().appendingPathComponent("FastaiNotebooks", isDirectory: true)
#      .appendingPathComponent("Sources", isDirectory: true)
#      .appendingPathComponent("FastaiNotebooks", isDirectory: true).appendingPathComponent(last)
#      .deletingPathExtension().appendingPathExtension("swift"))
# ```

# This function parses the underlying json behind a notebook to keep the code in the cells marked with `//export`.

//export
public func notebookToScript(fname: Path){
    let newname = fname.basename(dropExtension: true)+".swift"
    let url = fname.parent/"FastaiNotebooks/Sources/FastaiNotebooks"/newname
    do {
        let data = try Data(contentsOf: fname.url)
        let jsonData = try JSONSerialization.jsonObject(with: data, options: .allowFragments) as! [String: Any]
        let cells = jsonData["cells"] as! [[String:Any]]
        var module = """
/*
THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
file to edit: \(fname.lastPathComponent)

*/
        
"""
        for cell in cells {
            if let source = cell["source"] as? [String], !source.isEmpty, 
                   source[0].hasMatch(pat: #"^\s*//\s*export\s*$"#) {
                module.append("\n" + source[1...].joined() + "\n")
            }
        }
        try module.write(to: url, encoding: .utf8)
    } catch {
        print("Can't read the content of \(fname)")
    }
}

# And this will do all the notebooks in a given folder.

// export
public func exportNotebooks(_ path: Path) {
    for entry in try! path.ls()
    where entry.kind == Entry.Kind.file && 
          entry.path.basename().hasMatch(pat: #"^\d*_.*ipynb$"#) {
        print("Converting \(entry)")
        notebookToScript(fname: entry.path)
    }
}

notebookToScript(fname: Path.cwd/"00_load_data.ipynb")

# But now that we implemented it from scratch we're allowed to use it as a package ;). NotebookExport has been written by pcuenq
# and will make our lives easier.

import NotebookExport
let exporter = NotebookExport(Path.cwd/"00_load_data.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


