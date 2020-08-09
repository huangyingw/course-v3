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

# ## OpenCV Integration Example
# Note: SwiftCV package requires OpenCV installed in order to compile.

# Uncomment line below when using Colab (this installs OpenCV4)
# %system SwiftCV/install/install_colab.sh
%install-location $cwd/swift-install
%install '.package(path: "$cwd/SwiftCV")' SwiftCV
%install '.package(path: "$cwd/FastaiNotebook_08_data_block")' FastaiNotebook_08_data_block

# ### Imports

%include "EnableIPythonDisplay.swift"
import Foundation
import SwiftCV
import Path

import FastaiNotebook_08_data_block

// display opencv version
print(cvVersion())

# ### Load image

func readImage(_ path:String)->Mat {
    let cvImg = imread(path)
    return cvtColor(cvImg, nil, ColorConversionCode.COLOR_BGR2RGB)
}

let path = downloadImagenette(sz:"")
let allNames = fetchFiles(path: path/"train/n03425413", recurse: false, extensions: ["jpeg", "jpg"])
let fNames = Array(allNames[0..<256])
let ns = fNames.map {$0.string}
let imgpath = ns[2]
var cvImg = readImage(imgpath)

# ### Timing

cvImg.size

print(type(of:cvImg.dataPtr))

# [next slide](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g512a2e238a_144_0)

let ptr = UnsafeRawPointer(cvImg.dataPtr).assumingMemoryBound(to: UInt8.self)

ptr[2]

time(repeating:10) {_ = readImage(imgpath)}

cvImg.rows

import Python
import TensorFlow

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
IPythonDisplay.shell.enable_matplotlib("inline")

func show_img(_ img: Mat, _ w: Int = 7, _ h: Int = 5) {
    show_img(Tensor<UInt8>(cvMat: img)!, w, h)
}

show_img(cvImg)

time(repeating:10) {_ = resize(cvImg, nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_NEAREST)}

time(repeating:10) {_ = resize(cvImg, nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_LINEAR)}

time(repeating:10) {_ = resize(cvImg, nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_CUBIC)}

time(repeating:10) {_ = resize(cvImg, nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_AREA)}

cvImg = resize(cvImg, nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_CUBIC)

func readResized(_ fn:String)->Mat {
    return resize(readImage(fn), nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_CUBIC)
}

var imgs = ns[0..<10].map(readResized)

time(repeating:10) {_ = readResized(imgpath)}

# +
public protocol Countable {
    var count:Int {get}
}
extension Mat  :Countable {}
extension Array:Countable {}

public extension Sequence where Element:Countable {
    var totalCount:Int { return map{ $0.count }.reduce(0, +) }
}
# -

func collateMats(_ imgs:[Mat])->Tensor<Float> {
    let c = imgs.totalCount
    let ptr = UnsafeMutableRawPointer.allocate(byteCount: c, alignment: 1)
    defer {ptr.deallocate()}
    var p = ptr
    for img in imgs {
        p.copyMemory(from: img.dataPtr, byteCount: img.count)
        p += img.count
    }
    let r = UnsafeBufferPointer(start: ptr.bindMemory(to: UInt8.self, capacity: c), count: c)
    cvImg = imgs[0]
    let shape = TensorShape([imgs.count, cvImg.rows, cvImg.cols, cvImg.channels])
    let res = Tensor(shape: shape, scalars: r)
    return Tensor<Float>(res)/255.0
}

var t = collateMats(imgs)

t.shape

show_img(t[2])

time(repeating:10) {_ = collateMats(imgs)}

time { _ = ns.map(readResized) }

# ### OpenCV Transformations

# #### Resize

show_img(
    resize(cvImg, nil, Size(100, 50), 0, 0, InterpolationFlag.INTER_AREA)
)

# #### Zoom / Crop

let zoomMat = getRotationMatrix2D(Size(cvImg.cols, cvImg.rows / 2), 0, 1)
show_img(
    warpAffine(cvImg, nil, zoomMat, Size(600, 600))
)

# #### Rotate

let rotMat = getRotationMatrix2D(Size(cvImg.cols / 2, cvImg.rows / 2), 20, 1)
show_img(
    warpAffine(cvImg, nil, rotMat, Size(cvImg.cols, cvImg.rows))
)

# #### Pad

show_img(
    copyMakeBorder(cvImg, nil, 40, 40, 40, 40, BorderType.BORDER_CONSTANT, RGBA(0, 127, 0, 0))
)

# #### Blur

show_img(
    GaussianBlur(cvImg, nil, Size(25, 25))
)

# #### Flip

show_img(
    flip(cvImg, nil, FlipMode.HORIZONTAL)
)

# #### Transpose

show_img(
    transpose(cvImg, nil)
)


