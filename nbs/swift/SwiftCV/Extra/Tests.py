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

%install '.package(url: "https://github.com/vvmnnnkv/SwiftCV.git", .branch("master"))' SwiftCV

# ### Imports

# +
%include "EnableIPythonDisplay.swift"
import Foundation
import Python
import TensorFlow
import SwiftCV

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
IPythonDisplay.shell.enable_matplotlib("inline")
# -

// display opencv version
print(cvVersion())

# ### Utility

# +
func show_img(_ img: Mat, _ w: Int = 7, _ h: Int = 5) {
    // convert from OpenCV to Tensor
    let tens = Tensor<UInt8>(cvMat: img)!
    // and from Tensor to numpy array for matplot
    show_img(tens.makeNumpyArray(), w, h)
}

func show_img(_ img: PythonObject, _ w: Int = 7, _ h: Int = 5) {
    plt.figure(figsize: [w, h])
    plt.imshow(img)
    plt.show()
}
# -

# ### Load image

# +
// load image in memory
let url = "https://live.staticflickr.com/2842/11335865374_0b202e2dc6_o_d.jpg"
let imgContent = Data(contentsOf: URL(string: url)!)

// make opencv image
var cvImg = imdecode(imgContent)
// convert color scheme to RGB
cvImg = cvtColor(cvImg, nil, ColorConversionCode.COLOR_BGR2RGB)
show_img(cvImg)
# -

# ### OpenCV Transformations

# #### Resize

show_img(
    resize(cvImg, nil, Size(100, 50), 0, 0, InterpolationFlag.INTER_AREA)
)


# #### Zoom / Crop

let zoomMat = getRotationMatrix2D(Size(cvImg.cols, cvImg.rows / 2), 0, 2)
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

# ### Native S4TF Tensor Operations

# #### Lightning / Contrast

// convert image to floats Tensor
var imgTens = Tensor<Float>(Tensor<UInt8>(cvMat: cvImg)!) / 255
let contr:Float = 1.8
let lightn:Float = 0.2
let mean = imgTens.mean()
imgTens = (imgTens - mean) * contr + mean + lightn
show_img(imgTens.makeNumpyArray())

# #### Noise

# +
// convert image to Tensor
let smallImg = resize(cvImg, nil, Size(150, 150))
var imgTens = Tensor<Float>(Tensor<UInt8>(cvMat: smallImg)!) / 255

// make white noise (slow! :))
var rng = PhiloxRandomNumberGenerator(seed: UInt64(42))
let dist = NormalDistribution<Float>(mean: 0, standardDeviation: 0.05)
var random: [Float] = []
for _ in 0..<imgTens.shape.contiguousSize {
    random.append(dist.next(using: &rng))
}
let randTens = Tensor<Float>(shape: imgTens.shape, scalars: random)

imgTens += randTens
show_img(imgTens.makeNumpyArray())
# -


