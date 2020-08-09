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

# # Automatic Differentiation In Swift
#
# This notebook builds up the concepts of automatic differentiation in Swift from the constituent pieces.

# ## Step 1: A Trivial Example

# As a warm up, we will start with a trivial example $x^2$. The derivative $\frac{d}{dx} x^2$ is $2x$. We can represent this as follows in code.

# +
func square(_ x: Float) -> Float {
    return x * x
}

func square_derivative(_ x: Float) -> Float {
    return 2 * x
}
# -

# ## Aside: Recall the Chain Rule

# As we discussed before, the chain rule tells us how to differentiate composite functions, and is written: $$\frac{d}{dx}\left[f\left(g(x)\right)\right] = f'\left(g(x)\right)g'(x)$$

# ## Step 2: A slightly more complicated example.

# Simple polynomials are the easy case. Let's take the derivative of a more complicated function: $\sin(x^2)$.
#
# The derivative of this expression $\frac{d}{dx}\sin(x^2)$ (recall the chain rule) is: $\cos(x^2) \cdot 2x$.
#
# In code, this is as follows:

# +
import Glibc

func exampleFunction(_ x: Float) -> Float {
    return sin(x * x)
}

func exampleFunctionDerivative(_ x: Float) -> Float {
    return 2 * x * cos(x * x)
}
# -

# ## Step 3: A more efficient implementation

# Looking at the chain rule and our derivative implementation above, we notice that there's redundant computation going on. Concretely, in both `exampleFunction` and `exampleFunctionDerivative` we compute `x * x`. (In the chain rule definition, this is $g(x)$.) As a result, we often want to do that computation only once (because in general it can be any expensive computation, and even a multiply can be expensive with large tensors).
#
# We can thus rewrite our function and its derivative as follows:

func exampleFunctionDerivativeEfficient(_ x: Float) -> (value: Float, backward: () -> Float) {
    let xSquared = x * x
    let value = sin(xSquared)
    let backward = {2 * x * cos(xSquared)}  // A closure that captures xSquared
    return (value: value, backward: backward)
}

# You'll see that we're defining a somewhat more complex *closure* than we've seen before here.

# ## Aside: Fully general derivatives

# We've actually been a little sloppy with our mathematics. To be fully correct, $\frac{d}{dx}x^2 = 2x\frac{d}{dx}$. This is because in mathematics, $x$ doesn't have to be a specific number, it could be itself another expression, which we'd need to use the chain rule to calculate. In order to represent this correctly in code, we need to change the type signature slightly to multiply by the "$\frac{d}{dx}$", resulting in the following (we're changing the name `backward` to `deriv` here to signify that it's a little different):

func exampleFunctionValueWithDeriv(_ x: Float) -> (value: Float, deriv: (Float) -> Float) {
    let xSquared = x * x
    let value = sin(xSquared)
    let deriv = { (v: Float) -> Float in
        let gradXSquared = v * cos(xSquared)
        let gradX = gradXSquared * 2 * x
        return gradX
    }
    return (value: value, deriv: deriv)
}

# ## Step 4: Rewrite using `deriv`s

# We've chosen to represent the drivatives with a `deriv` closure because this allows us to rewrite the forward pass into a very regular form. Below, we rewrite the handwritten derivative above into a regular form.
#
# > Note: be sure to carefully read through the code and convince yourself that this new spelling of the `deriv` results in the exact same computation.

# +
func sinValueWithDeriv(_ x: Float) -> (value: Float, deriv: (Float) -> Float) {
    return (value: sin(x), deriv: {v in cos(x) * v})
}

func squareValueWithDeriv(_ x: Float) -> (value: Float, deriv: (Float) -> Float) {
    return (value: x * x, deriv: {v in 2 * x * v})
}

func exampleFunctionWithDeriv(_ x: Float) -> (value: Float, deriv: (Float) -> Float) {
    let (xSquared, deriv1) = squareValueWithDeriv(x)
    let (value, deriv2) = sinValueWithDeriv(xSquared)
    return (value: value, deriv: { v in
        let gradXSquared = deriv2(v)
        let gradX = deriv1(gradXSquared)
        return gradX
    })
}
# -

# ## Aside: Generalizing to arbitrary expressions.

# Up until this point, we've been handwriting the derivatives for specific functions. But we now have a formulation that is regular and composible. (In fact, it is so regular, we can make the computer write the backwards pass for us! aka automatic differentiation.) The rules are:
#
#  1. Rewrite every expression in the forward pass into a form that computes the value like normal, and also produces an additional deriv function.
#  2. Construct a backwards pass that threads the derivs together in the reverse order.

# In an abstract form, we transform a function that looks like:
#
# ```swift
# func myFunction(_ arg: Float) -> Float {
#     let tmp1 = expression1(arg)
#     let tmp2 = expression2(tmp1)
#     let tmp3 = expression3(tmp2)
#     return tmp3
# }
# ```

# into a function that looks like this:
#
# ```swift
# func myFunctionValueWithDeriv(_ arg: Float) -> (value: Float, deriv: (Float) -> Float) {
#     let (tmp1, deriv1) = expression1ValueWithDeriv(arg)
#     let (tmp2, deriv2) = expression2ValueWithDeriv(tmp1)
#     let (tmp3, deriv3) = expression3ValueWithDeriv(tmp2)
#     return (value: tmp3,
#             deriv: { v in
#                 let grad2 = deriv3(v)
#                 let grad1 = deriv2(grad2)
#                 let gradArg = deriv1(grad1)
#                 return gradArg
#     })
# }
# ```

# ## Step 5: Generalize beyond unary functions

# Up until now, we have been using functions that don't "reuse" values in the forward pass. Our running example of $\frac{d}{dx}\sin(x^2)$ is too simple. Let's make it a bit more complicated, and use $\frac{d}{dx}\sin(x^2)+x^2$ as our motivating expression going forward. From mathematics, we know that the derivative should be: $$\frac{d}{dx}\sin\left(x^2\right) + x^2 = \left(2x\cos\left(x^2\right)+2x\right)\frac{d}{dx}$$
#
# Let's see how we write the deriv (pay attention to the signature of the deriv for the `+` function)!

# +
func myComplexFunction(_ x: Float) -> Float {
    let tmp1 = square(x)
    let tmp2 = sin(tmp1)
    let tmp3 = tmp2 + tmp1
    return tmp3
}

func plusWithDeriv(_ x: Float, _ y: Float) -> (value: Float, deriv: (Float) -> (Float, Float)) {
    return (value: x + y, deriv: {v in (v, v)})  // Value semantics are great! :-)
}
# -

func myComplexFunctionValueWithDeriv(_ x: Float) -> (value: Float, deriv: (Float) -> Float) {
    let (tmp1, pb1) = squareValueWithDeriv(x)
    let (tmp2, pb2) = sinValueWithDeriv(tmp1)
    let (tmp3, pb3) = plusWithDeriv(tmp2, tmp1)
    return (value: tmp3,
            deriv: { v in
        // Initialize the gradients for all values at zero.
        var gradX = Float(0.0)
        var grad1 = Float(0.0)
        var grad2 = Float(0.0)
        var grad3 = Float(0.0)
        // Add the temporaries to the gradients as we run the backwards pass.
        grad3 += v
        let (tmp2, tmp1b) = pb3(grad3)
        grad2 += tmp2
        grad1 += tmp1b
        let tmp1a = pb2(grad2)
        grad1 += tmp1a
        let tmpX = pb1(grad1)
        gradX += tmpX
        // Return the computed gradients.
        return gradX
    })
}

// Helper method
func square(_ x: Float) -> Float {
    return x * x
}

# Non-unary functions (e.g. `+`) have a deriv that returns a tuple that corresponds to their arguments. This allows gradients to flow upwards in a pure functional manner.
#
# In order to handle the re-use of intermediary values (in this case, the expression $x^2$), we need to introduce 2 additional concepts:
#
#  1. **Sum**: We need to sum the derivatives produced by $\frac{d}{dx}x^2$ to the values produced from $\frac{d}{dx}\sin\left(x^2\right)$ in order to correctly compute the derivative value of $\frac{d}{dx}\left(\sin\left(x^2\right) + x^2\right)$.
#  2. **Zero**: As a result, we need to initialize the derivatives for each variable to a value: zero!
#
# We now have all the pieces required for automatic differentiation in Swift. Let's see how they come together.

# ## Step 6: Automatic Differentiation in Swift
#
# When you annotate a function `@differentiable`, the compiler will take your function and generate a second function that corresponds to the `...ValueWithDeriv` functions we wrote out by hand above using the simple transformation rules.
#
# You can access these auto-generated function by calling `valueWithPullback`:

@differentiable
func myFunction(_ x: Float) -> Float {
    return x * x
}

let (value, deriv) = valueWithPullback(at: 3, in: myFunction)
print(value)
print(type(of: deriv))

# ## Step 7: gradient
#
# Now that we have a deriv, how to we "kick off" the deriv computation to actually compute the derivative? We use the constant value `1.0`!

deriv(1)

# We have no re-implemented the `gradient` function.

# ## Step 8: Generalized Differentiability & Protocols

# So far, we've been looking at functions operating on scalar (`Float`) values, but you can take derivatives of functions that operate on vectors (aka higher dimensions) too. In order to support this, you need your type to conform to the `Differentiable` protocol, which often involves ensuring your type conforms to the [`AdditiveArithmetic` protocol](https://github.com/apple/swift/blob/0c452616820bfbc4f3197dd418c74adadc830b5c/stdlib/public/core/Integers.swift#L31). The salient bits of that protocol are:

# ```swift
# public protocol AdditiveArithmetic : Equatable {
#   /// The zero value.
#   ///
#   /// - Note: Zero is the identity element for addition; for any value,
#   ///   `x + .zero == x` and `.zero + x == x`.
#   static var zero: Self { get }
#   /// Adds two values and produces their sum.
#   ///
#   /// - Parameters:
#   ///   - lhs: The first value to add.
#   ///   - rhs: The second value to add.
#   static func +(lhs: Self, rhs: Self) -> Self
#   
#   //...
# }
# ```
#
# > Note: The [`Differentiable`](https://github.com/apple/swift/blob/0c452616820bfbc4f3197dd418c74adadc830b5c/stdlib/public/core/AutoDiff.swift#L102) protocol is slightly more complicated in order to support non-differentiable member variables, such as  activation functions and other non-differentiable member variables.

# ### Next up: The `Layer` protocol!
