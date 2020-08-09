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
%install '.package(path: "$cwd/FastaiNotebook_00_load_data")' FastaiNotebook_00_load_data

//export
import Path
import TensorFlow

import FastaiNotebook_00_load_data

# ## Get some Tensors to play with

# We can initialize a tensor in lots of different ways because in swift, two functions with the same name can coexist as long as they don't have the same signatures. Different named arguments give different signatures, so all of those are different `init` functions of `Tensor`:

let zeros = Tensor<Float>(zeros: [1,4,5])
let ones  = Tensor<Float>(ones: [12,4,5])
let twos  = Tensor<Float>(repeating: 2.0, shape: [2,3,4,5])
let range = Tensor<Int32>(rangeFrom: 0, to: 32, stride: 1)

# Those are just some examples and there are many more! Here we grab random numbers

let xTrain = Tensor<Float>(randomNormal: [5, 784])
var weights = Tensor<Float>(randomNormal: [784, 10]) / sqrt(784)
print(weights[0])

# # Building Matmul

# Ok, now that we know how floating point types and arrays work, we can finally build our own matmul from scratch, using a few loops.  We will take the two input matrices as single dimensional arrays so we can show manual indexing into them, the hard way:

// a and b are the flattened array elements, aDims/bDims are the #rows/columns of the arrays.
func swiftMatmul(a: [Float], b: [Float], aDims: (Int,Int), bDims: (Int,Int)) -> [Float] {
    assert(aDims.1 == bDims.0, "matmul shape mismatch")
    
    var res = Array(repeating: Float(0.0), count: aDims.0 * bDims.1)
    for i in 0 ..< aDims.0 {
        for j in 0 ..< bDims.1 {
            for k in 0 ..< aDims.1 {
                res[i*bDims.1+j] += a[i*aDims.1+k] * b[k*bDims.1+j]
            }
        }
    }
    return res
}

# To try this out, we extract the scalars out of our MNIST data as an array.

let flatA = xTrain[0..<5].scalars
let flatB = weights.scalars
let (aDims,bDims) = ((5, 784), (784, 10))

# Now that we've got everything together, we can try it out!

var resultArray = swiftMatmul(a: flatA, b: flatB, aDims: aDims, bDims: bDims)

time(repeating: 100) {
    _ = swiftMatmul(a: flatA, b: flatB, aDims: aDims, bDims: bDims)
}

# Awesome, that is pretty fast - compare that to **835 ms** with Python!
#
# You might be wondering what that `time(repeating:)` builtin is.  As you might guess, this is actually a Swift function - one that is using "trailing closure" syntax to specify the body of the timing block.  Trailing closures are passed as arguments to the function, and in this case, the function was defined in our ✅**00_load_data** workbook.  Let's take a look!
#

# # Getting the performance of C 💯

# This performance is pretty great, but we can do better.  Swift is a memory safe language (like Python), which means it has to do array bounds checks and some other stuff.  Fortunately, Swift is a pragmatic language that allows you to drop through this to get peak performance - check out Jeremy's article [High Performance Numeric Programming with Swift: Explorations and Reflections](https://www.fast.ai/2019/01/10/swift-numerics/) for a deep dive.
#
# One thing you can do is use `UnsafePointer` (which is basically a raw C pointer) instead of using a bounds checked array.  This isn't memory safe, but gives us about a 2x speedup in this case!

// a and b are the flattened array elements, aDims/bDims are the #rows/columns of the arrays.
func swiftMatmulUnsafe(a: UnsafePointer<Float>, b: UnsafePointer<Float>, aDims: (Int,Int), bDims: (Int,Int)) -> [Float] {
    assert(aDims.1 == bDims.0, "matmul shape mismatch")
    
    var res = Array(repeating: Float(0.0), count: aDims.0 * bDims.1)
    res.withUnsafeMutableBufferPointer { res in 
        for i in 0 ..< aDims.0 {
            for j in 0 ..< bDims.1 {
                for k in 0 ..< aDims.1 {
                    res[i*bDims.1+j] += a[i*aDims.1+k] * b[k*bDims.1+j]
                }
            }
        }
    }
    return res
}

time(repeating: 100) {
    _ = swiftMatmulUnsafe(a: flatA, b: flatB, aDims: aDims, bDims: bDims)
}

# One of the other cool things about this is that we can provide a nice idiomatic API to the caller of this, and keep all the unsafe shenanigans inside the implementation of this function.
#
# If you really want to fall down the rabbit hole, you can look at the [implementation of `UnsafePointer`](https://github.com/apple/swift/blob/tensorflow/stdlib/public/core/UnsafePointer.swift), which is of written in Swift wrapping LLVM pointer operations.  This means you can literally get the performance of C code directly in Swift, while providing easy to use high level APIs!

# ## Swift 💖 C APIs too: you get the full utility of the C ecosystem

# Swift even lets you transparently work with C APIs, just like it does with Python.  This can be used for both good and evil.  For example, here we directly call the `malloc` function, dereference the uninitialized pointer, and print it out:

# +
import Glibc

let ptr : UnsafeMutableRawPointer = malloc(42)

print("☠️☠️ Uninitialized garbage =", ptr.load(as: UInt8.self))

free(ptr)
# -

# An `UnsafeMutableRawPointer` ([implementation](https://github.com/apple/swift/blob/tensorflow/stdlib/public/core/UnsafeRawPointer.swift)) isn't something you should use lightly, but when you work with C APIs, you'll see various types like that in the function signatures.
#
# Calling `malloc` and `free` directly aren't recommended in Swift, but is useful and important when you're working with C APIs that expect to get malloc'd memory, which comes up when you're written a safe Swift wrapper for some existing code.
#
# Speaking of existing code, let's take a look at that **Python interop** we touched on before:
#

# +
import Python
let np = Python.import("numpy")
let pickle = Python.import("pickle")
let sys = Python.import("sys")

print("🐍list =    ", pickle.dumps([1, 2, 3]))
print("🐍ndarray = ", pickle.dumps(np.array([[1, 2], [3, 4]])))

# -

# Of course this is [all written in Swift](https://github.com/apple/swift/tree/tensorflow/stdlib/public/Python) as well.  You can probably guess how this works now: `PythonObject` is a Swift struct that wraps a pointer to the Python interpreter's notion of a Python object.
#
# ```swift
# @dynamicCallable
# @dynamicMemberLookup
# public struct PythonObject {
#   var reference: PyReference
#   ...
# }
# ```
#
# The `@dynamicMemberLookup` attribute allows it to dynamically handle all member lookups (like `x.y`) by calling into [the `PyObject_GetAttrString` runtime call](https://github.com/apple/swift/blob/tensorflow/stdlib/public/Python/Python.swift#L427).  Similarly, the `@dynamicCallable` attribute allows the type to intercept all calls to a PythonObject (like `x()`), which it implements using the [`PyObject_Call` runtime call](https://github.com/apple/swift/blob/tensorflow/stdlib/public/Python/Python.swift#L324).  
#
# Because Swift has such simple and transparent access to C, it allows building very nice first-class Swift APIs that talk directly to the lower level implementation, and these implementations can have very little overhead.

# # Working with Tensor

# Lets get back into matmul and explore more of the `Tensor` type as provided by the TensorFlow module.  You can see all things `Tensor` can do in the [official documentation](https://www.tensorflow.org/swift/api_docs/Structs/Tensor).
#
# Here are some highlights. We saw how you can get zeros or random data:

# +
var bias = Tensor<Float>(zeros: [10])

let m1 = Tensor<Float>(randomNormal: [5, 784])
let m2 = Tensor<Float>(randomNormal: [784, 10])
# -

# Tensors carry data and a shape.

print("m1: ", m1.shape)
print("m2: ", m2.shape)

# The `Tensor` type provides all the normal stuff you'd expect as methods.  Including arithmetic, convolutions, etc and this includes full support for broadcasting:
#

# +
let small = Tensor<Float>([[1, 2],
                           [3, 4]])

print("🔢2x2:\n", small)
# -

# **MatMul Operator:** In addition to using the global `matmul(a, b)` function, you can also use the `a • b` operator to matmul together two things.  This is just like the `@` operator in Python.  You can get it with the <kbd>option</kbd>-<kbd>8</kbd> on Mac or <kbd>compose</kbd>-<kbd>.</kbd>-<kbd>=</kbd> elsewhere. Or if you prefer, just use the `matmul()` function we've seen already.

print("⊞ matmul:\n",  matmul(small, small))
print("\n⊞ again:\n", small • small)


# **Reshaping** works the way you'd expect:

var m = Tensor([1.0, 2, 3, 4, 5, 6, 7, 8, 9]).reshaped(to: [3, 3])
print(m)

# You have the basic mathematical functions:

sqrt((m * m).sum())

# ## Elementwise ops and comparisons

# Standard math operators (`+`,`-`,`*`,`/`) are all element-wise, and there are a bunch of standard math functions like `sqrt` and `pow`.  Here are some examples:

var a = Tensor([10.0, 6, -4])
var b = Tensor([2.0, 8, 7])
(a,b)

print("add:  ", a + b)
print("mul:  ", a * b)
print("sqrt: ", sqrt(a))
print("pow:  ", pow(a, b))

#
# **Comparison operators** (`>`,`<`,`==`,`!=`,...) in Swift are supposed to return a single `Bool` value, so they are `true` if all the elements of the tensors satisfy the comparison.
#
# Elementwise versions have the `.` prefix, which is read as "pointwise": `.>`, `.<`, `.==`, etc.  You can merge a tensor of bools into a single Bool with the `any()` and `all()` methods.

a < b

a .< b

print((a .> 0).all())

print((a .> 0).any())

# ## Broadcasting

# Broadcasting with a scalar works just like in Python:

var a = Tensor([10.0, 6.0, -4.0])

print(a+1)

2 * m

# #### Broadcasting a vector with a matrix

let c = Tensor([10.0,20.0,30.0])

# By default, broadcasting is done by adding 1 dimensions to the beginning until dimensions of both objects match.

m + c

c + m

# To broadcast on the other dimensions, one has to use `expandingShape` to add the dimension.

m + c.expandingShape(at: 1)

c.expandingShape(at: 1)

# #### Broadcasting rules

print(c.expandingShape(at: 0).shape)

print(c.expandingShape(at: 1).shape)

c.expandingShape(at: 0) * c.expandingShape(at: 1)

c.expandingShape(at: 0) .> c.expandingShape(at: 1)

# # Matmul using `Tensor`
#
# Coming back to our matmul algorithm, we can implement exactly what we had before by using subscripting into a tensor, instead of subscripting into an array.  Let's see how that works:

# +
func tensorMatmul(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Tensor<Float> {
    var res = Tensor<Float>(zeros: [a.shape[0], b.shape[1]])

    for i in 0 ..< a.shape[0] {
        for j in 0 ..< b.shape[1] {
            for k in 0 ..< a.shape[1] {
                res[i, j] += a[i, k] * b[k, j]
            }
        }
    }
    return res
}

_ = tensorMatmul(m1, m2)
# -

time { 
    let tmp = tensorMatmul(m1, m2)
    
    // Copy a scalar back to the host to force a GPU sync.
    _ = tmp[0, 0].scalar
}

# What, what just happened?? We used to be less than a **tenth of a millisecond**, now we're taking **multiple seconds**.  It turns out that Tensor's are very good at bulk data processing, but they are not good at doing one float at a time.  Make sure to use the coarse-grained operations.  We can make this faster by vectorizing each loop in turn.
#
# **Slides:** [Granularity of Tensor Operations](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g58253914c1_0_380). 

# ## Vectorize the inner loop into a multiply + sum

# +
func elementWiseMatmul(_ a:Tensor<Float>, _ b:Tensor<Float>) -> Tensor<Float>{
    let (ar, ac) = (a.shape[0], a.shape[1])
    let (br, bc) = (b.shape[0], b.shape[1])
    var res = Tensor<Float>(zeros: [ac, br])
    
    for i in 0 ..< ar {
        let row = a[i]
        for j in 0 ..< bc {
            res[i, j] = (row * b.slice(lowerBounds: [0,j], upperBounds: [ac,j+1]).squeezingShape(at: 1)).sum()
        }
    }
    return res
}

_ = elementWiseMatmul(m1, m2)
# -

time { 
    let tmp = elementWiseMatmul(m1, m2)

    // Copy a scalar back to the host to force a GPU sync.
    _ = tmp[0, 0].scalar
}

# ## Vectorize the inner two loops with broadcasting

# +
func broadcastMatmult(_ a:Tensor<Float>, _ b:Tensor<Float>) -> Tensor<Float>{
    var res = Tensor<Float>(zeros: [a.shape[0], b.shape[1]])
    for i in 0..<a.shape[0] {
        res[i] = (a[i].expandingShape(at: 1) * b).sum(squeezingAxes: 0)
    }
    return res
}

_ = broadcastMatmult(m1, m2)
# -

time(repeating: 100) {
    let tmp = broadcastMatmult(m1, m2)

    // Copy a scalar back to the host to force a GPU sync.
    _ = tmp[0, 0].scalar
}

# ## Vectorize the whole thing with one Tensorflow op

time(repeating: 100) { _ = m1 • m2 }

# Ok, now that we have matmul, we can continue to build out our framework.
#
# To complete today's lesson, let's jump way way up the stack to see **Workbook 11**.
#
#
#
# ## Tensorflow vectorizes, parallelizes, and scales
#
# The reason that TensorFlow works in practice is that it can scale way up to large matrices, for example, lets try some thing a bit larger:

# +
func timeMatmulTensor(size: Int) {
    var matrix = Tensor<Float>(randomNormal: [size, size])
    print("\n\(size)x\(size):\n  ⏰", terminator: "")
    time(repeating: 10) { 
        let matrix = matrix • matrix 
        _ = matrix[0, 0].scalar
    }
}

timeMatmulTensor(size: 1)     // Tiny
timeMatmulTensor(size: 10)    // Bigger
timeMatmulTensor(size: 100)   // Even Bigger
timeMatmulTensor(size: 1000)  // Biggerest
timeMatmulTensor(size: 5000)  // Even Biggerest
# -

# In constrast, our simple CPU implementation takes a lot longer to do the same work.  For example:

# +
func timeMatmulSwift(size: Int, repetitions: Int = 10) {
    var matrix = Tensor<Float>(randomNormal: [size, size])
    let matrixFlatArray = matrix.scalars

    print("\n\(size)x\(size):\n  ⏰", terminator: "")
    time(repeating: repetitions) { 
       _ = swiftMatmulUnsafe(a: matrixFlatArray, b: matrixFlatArray, aDims: (size,size), bDims: (size,size))
    }
}

timeMatmulSwift(size: 1)     // Tiny
timeMatmulSwift(size: 10)    // Bigger
timeMatmulSwift(size: 100)   // Even Bigger
timeMatmulSwift(size: 1000, repetitions: 1)  // Biggerest

print("\n5000x5000: skipped, it takes tooo long!")
# -

# Why is TensorFlow *so so so* much faster than our CPU implementation?  Well there are two reasons: the first of which is that it uses GPU hardware, which is much faster for math like this.  That said, there are a ton of tricks (involving memory hierarchies, cache blocking, and other tricks) that make matrix multiplications go fast on CPUs and other hardware.
#
# For example, try using TensorFlow on the CPU to do the same computation as above:

withDevice(.cpu) {
    timeMatmulTensor(size: 1)     // Tiny
    timeMatmulTensor(size: 10)    // Bigger
    timeMatmulTensor(size: 100)   // Even Bigger
    timeMatmulTensor(size: 1000)  // Biggerest
    timeMatmulTensor(size: 5000)  // Even Biggerest
}

# This is a pretty big difference.  On my hardware, it takes 2287ms for Swift to do a 1000x1000 multiply on the CPU, it takes TensorFlow 6.7ms to do the same work on the CPU, and takes TensorFlow 0.49ms to do it on a GPU.
#
# # Hardware Accelerators vs Flexibility
#
# One of the big challenges with machine learning frameworks today is that they provide a fixed set of "ops" that you can use with high performance.  There is a lot of work underway to fix this.  The [XLA compiler in TensorFlow](https://www.tensorflow.org/xla) is an important piece of this, which allows more flexibility in the programming model while still providing high performance by using compilers to target the hardware accelerator.  If you're interested in the details, there is a [great video by the creator of Halide](https://www.youtube.com/watch?v=3uiEyEKji0M) explaining why this is challenging.
#
# TensorFlow internals are undergoing [significant changes (slide)](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.g58253914c1_3_0) including the introduction of the XLA compiler, and the introduction of [MLIR compiler technology](https://github.com/tensorflow/mlir).
#
#
# # Tensor internals and Raw TensorFlow operations
#
# TensorFlow provides hundreds of different operators, and they sort of grew organically over time.  This means that there are some deprecated operators, they aren't particularly consistent, and there are other oddities.  As such, the `Tensor` type provides a curated set of these operators as methods.
#
# Whereas `Int` and `Float` are syntactic sugar for LLVM, and `PythonObject` is syntactic sugar for the Python interpreter, `Tensor` ends up being syntactic sugar for the TensorFlow operator set.  You can dive in and see its implementation in Swift in [the S4TF `TensorFlow` module](https://github.com/apple/swift/blob/tensorflow/stdlib/public/TensorFlow/Tensor.swift), e.g.:
#
# ```swift
# public struct Tensor<Scalar : TensorFlowScalar> : TensorProtocol {
#   /// The underlying `TensorHandle`.
#   /// - Note: `handle` is public to allow user defined ops, but should not
#   /// normally be used otherwise.
#   public let handle: TensorHandle<Scalar>
#   ... 
# }
# ```
#
# Here we see the internal implementation details of `Tensor`, which stores a `TensorHandle` - the internal implementation detail of the TensorFlow Eager runtime.
#
# Methods are defined on Tensor just like you'd expect, here [is the basic addition operator](https://github.com/apple/swift/blob/tensorflow/stdlib/public/TensorFlow/Ops.swift#L88), defined over all numeric tensors (i.e., not tensors of `Bool`):
#
# ```swift
# extension Tensor : AdditiveArithmetic where Scalar : Numeric {
#   /// Adds two tensors and produces their sum.
#   /// - Note: `+` supports broadcasting.
#   public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
#     return Raw.add(lhs, rhs)
#   }
# }
# ```
#
# But wait, what is this Raw thing?

# ### Raw TensorFlow ops
#
# TensorFlow has a database of the operators it defines, which gets encoded into a [protocol buffer](https://developers.google.com/protocol-buffers/).  From this protobuf, *all* of the operators automatically get a Raw operator (implemented in terms of a lower level `#tfop` primitive).
#

# +
// Explore the contents of the Raw namespace by typing Raw.<tab>
print(Raw.zerosLike(c))

// Raw.
# -

# There is an [entire tutorial on Raw operators](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/raw_tensorflow_operators.ipynb) on github/TensorFlow/swift.  The key thing to know is that TensorFlow can do almost anything, so if there is no obvious method on `Tensor` to do what you need it is worth checking out the tutorial to see how to do this.
#
# As one example, later parts of the tutorial need the ability to load files and decode JPEGs.  Swift for TensorFlow doesn't have these as methods on `StringTensor` yet, but we can add them like this:

//export
public extension StringTensor {
    // Read a file into a Tensor.
    init(readFile filename: String) {
        self.init(readFile: StringTensor(filename))
    }
    init(readFile filename: StringTensor) {
        self = Raw.readFile(filename: filename)
    }

    // Decode a StringTensor holding a JPEG file into a Tensor<UInt8>.
    func decodeJpeg(channels: Int = 0) -> Tensor<UInt8> {
        return Raw.decodeJpeg(contents: self, channels: Int64(channels), dctMethod: "") 
    }
}


# ### Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"01_matmul.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


