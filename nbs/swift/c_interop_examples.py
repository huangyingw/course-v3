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

# ## C Integration Examples
#
# Notes:
#
# - SwiftSox package requires sox to be installed: `sudo apt install libsox-dev libsox-fmt-all sox`
# - SwiftVips package requires vips to be installed: see `SwiftVips/install.sh` for steps

%install-extra-include-command pkg-config --cflags vips
%install-location $cwd/swift-install
%install '.package(path: "$cwd/SwiftVips")' SwiftVips
%install '.package(path: "$cwd/SwiftSox")' SwiftSox
%install '.package(path: "$cwd/FastaiNotebook_08_data_block")' FastaiNotebook_08_data_block

import Foundation
import Path
import FastaiNotebook_08_data_block

# ### Sox

import sox

# +
public func InitSox() {
  if sox_format_init() != SOX_SUCCESS.rawValue { fatalError("Can not init SOX!") }
}

public func ReadSoxAudio(_ name:String)->UnsafeMutablePointer<sox_format_t> {
  return sox_open_read(name, nil, nil, nil)
}
# -

InitSox()

let fd = ReadSoxAudio("SwiftSox/sounds/chris.mp3")

let sig = fd.pointee.signal

(sig.rate,sig.precision,sig.channels,sig.length)

var samples = [Int32](repeating: 0, count: numericCast(sig.length))

sox_read(fd, &samples, numericCast(sig.length))

import Python

%include "EnableIPythonDisplay.swift"
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let display = Python.import("IPython.display")
IPythonDisplay.shell.enable_matplotlib("inline")

let t = samples.makeNumpyArray()

plt.figure(figsize: [12, 4])
plt.plot(t[2000..<4000])
plt.show()

display.Audio(t, rate:sig.rate).display()

# So here we're using numpy, matplotlib, ipython, all from swift! 😎
#
# Why limit ourselves to Python? There's a lot out there that's not in Python yet!

# [next slide](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g512a2e238a_144_16)

# ### Vips

import TensorFlow
import SwiftVips
import CSwiftVips
import vips

vipsInit()

let path = downloadImagenette()
let allNames = fetchFiles(path: path/"train", recurse: true, extensions: ["jpeg", "jpg"])
let fNames = Array(allNames[0..<256])
let ns = fNames.map {$0.string}

let imgpath = ns[0]
let img = vipsLoadImage(imgpath)!

func vipsToTensor(_ img:Image)->Tensor<UInt8> {
    var sz = 0
    let mem = vipsGet(img, &sz)
    defer {free(mem)}
    let shape = TensorShape(vipsShape(img))
    return Tensor(shape: shape, scalars: UnsafeBufferPointer(start: mem, count: sz))
}

show_img(vipsToTensor(img))

# ## fin


