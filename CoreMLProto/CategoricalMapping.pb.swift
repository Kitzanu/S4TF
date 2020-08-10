// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: coremltools-master/mlmodel/format/CategoricalMapping.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///*
/// A categorical mapping.
///
/// This allows conversion from integers to strings, or from strings to integers.
struct CoreML_Specification_CategoricalMapping {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var mappingType: CoreML_Specification_CategoricalMapping.OneOf_MappingType? = nil

  /// Conversion from strings to integers
  var stringToInt64Map: CoreML_Specification_StringToInt64Map {
    get {
      if case .stringToInt64Map(let v)? = mappingType {return v}
      return CoreML_Specification_StringToInt64Map()
    }
    set {mappingType = .stringToInt64Map(newValue)}
  }

  /// Conversion from integer to string
  var int64ToStringMap: CoreML_Specification_Int64ToStringMap {
    get {
      if case .int64ToStringMap(let v)? = mappingType {return v}
      return CoreML_Specification_Int64ToStringMap()
    }
    set {mappingType = .int64ToStringMap(newValue)}
  }

  ///*
  /// The value returned if an input is not contained in the map above.
  /// If one of these is not set, then an error is raised on an unknown input.
  var valueOnUnknown: CoreML_Specification_CategoricalMapping.OneOf_ValueOnUnknown? = nil

  /// Default output when converting from an integer to a string.
  var strValue: String {
    get {
      if case .strValue(let v)? = valueOnUnknown {return v}
      return String()
    }
    set {valueOnUnknown = .strValue(newValue)}
  }

  /// Default output when converting from a string to an integer.
  var int64Value: Int64 {
    get {
      if case .int64Value(let v)? = valueOnUnknown {return v}
      return 0
    }
    set {valueOnUnknown = .int64Value(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  enum OneOf_MappingType: Equatable {
    /// Conversion from strings to integers
    case stringToInt64Map(CoreML_Specification_StringToInt64Map)
    /// Conversion from integer to string
    case int64ToStringMap(CoreML_Specification_Int64ToStringMap)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_CategoricalMapping.OneOf_MappingType, rhs: CoreML_Specification_CategoricalMapping.OneOf_MappingType) -> Bool {
      switch (lhs, rhs) {
      case (.stringToInt64Map(let l), .stringToInt64Map(let r)): return l == r
      case (.int64ToStringMap(let l), .int64ToStringMap(let r)): return l == r
      default: return false
      }
    }
  #endif
  }

  ///*
  /// The value returned if an input is not contained in the map above.
  /// If one of these is not set, then an error is raised on an unknown input.
  enum OneOf_ValueOnUnknown: Equatable {
    /// Default output when converting from an integer to a string.
    case strValue(String)
    /// Default output when converting from a string to an integer.
    case int64Value(Int64)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_CategoricalMapping.OneOf_ValueOnUnknown, rhs: CoreML_Specification_CategoricalMapping.OneOf_ValueOnUnknown) -> Bool {
      switch (lhs, rhs) {
      case (.strValue(let l), .strValue(let r)): return l == r
      case (.int64Value(let l), .int64Value(let r)): return l == r
      default: return false
      }
    }
  #endif
  }

  init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_CategoricalMapping: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".CategoricalMapping"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "stringToInt64Map"),
    2: .same(proto: "int64ToStringMap"),
    101: .same(proto: "strValue"),
    102: .same(proto: "int64Value"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1:
        var v: CoreML_Specification_StringToInt64Map?
        if let current = self.mappingType {
          try decoder.handleConflictingOneOf()
          if case .stringToInt64Map(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {self.mappingType = .stringToInt64Map(v)}
      case 2:
        var v: CoreML_Specification_Int64ToStringMap?
        if let current = self.mappingType {
          try decoder.handleConflictingOneOf()
          if case .int64ToStringMap(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {self.mappingType = .int64ToStringMap(v)}
      case 101:
        if self.valueOnUnknown != nil {try decoder.handleConflictingOneOf()}
        var v: String?
        try decoder.decodeSingularStringField(value: &v)
        if let v = v {self.valueOnUnknown = .strValue(v)}
      case 102:
        if self.valueOnUnknown != nil {try decoder.handleConflictingOneOf()}
        var v: Int64?
        try decoder.decodeSingularInt64Field(value: &v)
        if let v = v {self.valueOnUnknown = .int64Value(v)}
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    switch self.mappingType {
    case .stringToInt64Map(let v)?:
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    case .int64ToStringMap(let v)?:
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    case nil: break
    }
    switch self.valueOnUnknown {
    case .strValue(let v)?:
      try visitor.visitSingularStringField(value: v, fieldNumber: 101)
    case .int64Value(let v)?:
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 102)
    case nil: break
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_CategoricalMapping, rhs: CoreML_Specification_CategoricalMapping) -> Bool {
    if lhs.mappingType != rhs.mappingType {return false}
    if lhs.valueOnUnknown != rhs.valueOnUnknown {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
