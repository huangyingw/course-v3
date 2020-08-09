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
%install '.package(path: "$cwd/FastaiNotebook_08_data_block")' FastaiNotebook_08_data_block

//export
import Path

import FastaiNotebook_08_data_block

// export
public protocol HetDictKey {
    associatedtype ValueType
    static var defaultValue: ValueType { get }
}

# +
// export

public struct HeterogeneousDictionary {
    private var underlying: [ObjectIdentifier : Any] = [:]
    
    public init() {}
    public init<T: HetDictKey>(_ key: T.Type, _ value: T.ValueType) {
        self.underlying = [ObjectIdentifier(key): value]
    }
    public init<T1: HetDictKey, T2: HetDictKey>(_ key1: T1.Type, _ value1: T1.ValueType, _ key2: T2.Type, _ value2: T2.ValueType) {
        self.underlying = [ObjectIdentifier(key1): value1, ObjectIdentifier(key2): value2]
    }

    public subscript<T: HetDictKey>(key: T.Type) -> T.ValueType {
        get { return underlying[ObjectIdentifier(key), default: T.defaultValue] as! T.ValueType }
        set { underlying[ObjectIdentifier(key)] = newValue as Any }
    }
    
    public mutating func merge(_ other: HeterogeneousDictionary,
        uniquingKeysWith combine: (Any, Any) throws -> Any) rethrows {
        try self.underlying.merge(other.underlying, uniquingKeysWith: combine)
    }
}

# -

// export
// Common keys
public struct LearningRate: HetDictKey {
    public static var defaultValue: Float = 0.4
}

public struct StepCount: HetDictKey {
    public static var defaultValue = 0
}

// Sample usage
var m = HeterogeneousDictionary()


# +
print(m[LearningRate.self])
m[LearningRate.self] = 3.4
print(m[LearningRate.self])

print(m[StepCount.self])
m[StepCount.self] = 3
print(m[StepCount.self])

# -

print(type(of: m[StepCount.self]))
print(type(of: m[LearningRate.self]))


# ## Export

import NotebookExport
let exporter = NotebookExport(Path.cwd/"08a_heterogeneous_dictionary.ipynb")
print(exporter.export(usingPrefix: "FastaiNotebook_"))


