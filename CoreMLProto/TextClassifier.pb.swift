// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: coremltools-master/mlmodel/format/TextClassifier.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2018, Apple Inc. All rights reserved.
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
/// A model which takes a single input string and outputs a
/// label for the input.
struct CoreML_Specification_CoreMLModels_TextClassifier {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///
  /// Stores the resivion number for the model, revision 1 is available on
  /// iOS, tvOS 12.0+, macoOS 10.14+
  var revision: UInt32 = 0

  ///
  /// Stores the language of the model, as specified in BCP-47 format,
  /// e.g. "en-US". See https://tools.ietf.org/html/bcp47
  var language: String = String()

  ///
  /// Stores the byte representation of learned model parameters
  var modelParameterData: Data = SwiftProtobuf.Internal.emptyData

  ///
  /// Stores the set of output class labels
  var classLabels: CoreML_Specification_CoreMLModels_TextClassifier.OneOf_ClassLabels? = nil

  var stringClassLabels: CoreML_Specification_StringVector {
    get {
      if case .stringClassLabels(let v)? = classLabels {return v}
      return CoreML_Specification_StringVector()
    }
    set {classLabels = .stringClassLabels(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  ///
  /// Stores the set of output class labels
  enum OneOf_ClassLabels: Equatable {
    case stringClassLabels(CoreML_Specification_StringVector)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_CoreMLModels_TextClassifier.OneOf_ClassLabels, rhs: CoreML_Specification_CoreMLModels_TextClassifier.OneOf_ClassLabels) -> Bool {
      switch (lhs, rhs) {
      case (.stringClassLabels(let l), .stringClassLabels(let r)): return l == r
      }
    }
  #endif
  }

  init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification.CoreMLModels"

extension CoreML_Specification_CoreMLModels_TextClassifier: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".TextClassifier"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "revision"),
    10: .same(proto: "language"),
    100: .same(proto: "modelParameterData"),
    200: .same(proto: "stringClassLabels"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularUInt32Field(value: &self.revision)
      case 10: try decoder.decodeSingularStringField(value: &self.language)
      case 100: try decoder.decodeSingularBytesField(value: &self.modelParameterData)
      case 200:
        var v: CoreML_Specification_StringVector?
        if let current = self.classLabels {
          try decoder.handleConflictingOneOf()
          if case .stringClassLabels(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {self.classLabels = .stringClassLabels(v)}
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.revision != 0 {
      try visitor.visitSingularUInt32Field(value: self.revision, fieldNumber: 1)
    }
    if !self.language.isEmpty {
      try visitor.visitSingularStringField(value: self.language, fieldNumber: 10)
    }
    if !self.modelParameterData.isEmpty {
      try visitor.visitSingularBytesField(value: self.modelParameterData, fieldNumber: 100)
    }
    if case .stringClassLabels(let v)? = self.classLabels {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 200)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_CoreMLModels_TextClassifier, rhs: CoreML_Specification_CoreMLModels_TextClassifier) -> Bool {
    if lhs.revision != rhs.revision {return false}
    if lhs.language != rhs.language {return false}
    if lhs.modelParameterData != rhs.modelParameterData {return false}
    if lhs.classLabels != rhs.classLabels {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
