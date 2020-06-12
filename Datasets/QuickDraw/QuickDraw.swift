//QuickDraw/Datasets/QuickDraw/QuickDraw.swift

// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Original source:
// "The MNIST database of handwritten digits"
// Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
// http://yann.lecun.com/exdb/mnist/
import Foundation
import Batcher
import TensorFlow
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let np = Python.import("numpy")
let tf = Python.import("tensorflow.compat.v2")
let tfds = Python.import("tensorflow_datasets.public_api")

var _QUICKDRAW_IMAGE_SIZE = 28
var _QUICKDRAW_IMAGE_SHAPE = [_QUICKDRAW_IMAGE_SIZE, _QUICKDRAW_IMAGE_SIZE, 1]
var _QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"  // pylint: disable=line-too-long
var _QUICKDRAW_LABELS_FNAME = "image_classification/quickdraw_labels.txt"

public struct QuickDraw<Entropy: RandomNumberGenerator> {
  /// Type of the collection of non-collated batches.
  public typealias Batches = Slices<Sampling<[(data: [UInt8], label: Int32)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[(data: [UInt8], label: Int32)], Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  public typealias Validation = LazyMapSequence<Slices<[(data: [UInt8], label: Int32)]>, LabeledImage>
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation

  /// Creates an instance with `batchSize`.
  ///
  /// - Parameter entropy: a source of randomness used to shuffle sample 
  ///   ordering.  It  will be stored in `self`, so if it is only pseudorandom 
  ///   and has value semantics, the sequence of epochs is deterministic and not 
  ///   dependent on other operations.
  public init(batchSize: Int, entropy: Entropy, device: Device) {
    self.init(batchSize: batchSize, device: device, entropy: entropy,
              flattening: false, normalizing: false)
  }

  /// Creates an instance with `batchSize` on `device`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering.  It  
  ///     will be stored in `self`, so if it is only pseudorandom and has value 
  ///     semantics, the sequence of epochs is deterministic and not dependent 
  ///     on other operations.
  ///   - flattening: flattens the data to be a 2d-tensor iff `true. The default value
  ///     is `false`.
  ///   - normalizing: normalizes the batches to have values from -1.0 to 1.0 iff `true`.
  ///     The default value is `false`.
  ///   - localStorageDirectory: the directory in which the dataset is stored.
  public init(
    batchSize: Int, device: Device, entropy: Entropy, flattening: Bool = false, 
    normalizing: Bool = false, 
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory
      .appendingPathComponent("QuickDraw", isDirectory: true)
  ) {
    training = TrainingEpochs(
      samples: fetchQuickDrawDataset(
        //localStorageDirectory: localStorageDirectory,
        localStorageDirectory: swift-models/datasets/QuickDraw,
        remoteBaseDirectory: "https://vsod-my.sharepoint.com/personal/adrien_leroy_viseo_com",
        //remoteBaseDirectory: "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap",     
        //imagesFilename:  String(tfds.features.Image(shape:_QUICKDRAW_IMAGE_SHAPE))!,
        imagesFilename: "images_train",
        labelsFilename: "targets_train"),
        //labelsFilename:  String(tfds.features.ClassLabel(names_file:tfds.core.get_tfds_path(_QUICKDRAW_LABELS_FNAME)))!
      batchSize: batchSize, entropy: entropy
    ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
      return batches.lazy.map{ makeQuickDrawBatch(
        samples: $0, flattening: flattening, normalizing: normalizing, device: device
      )}
    }
    
    validation = fetchQuickDrawDataset(
      //localStorageDirectory: localStorageDirectory,
      localStorageDirectory: swift-models/datasets/QuickDraw,
      remoteBaseDirectory: "https://vsod-my.sharepoint.com/personal/adrien_leroy_viseo_com",
      //remoteBaseDirectory: "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap",
      //imagesFilename:  String(tfds.features.Image(shape:_QUICKDRAW_IMAGE_SHAPE))!,
      imagesFilename: "images_valid",
      labelsFilename: "targets_valid"
      //labelsFilename:  String(tfds.features.ClassLabel(names_file:tfds.core.get_tfds_path(_QUICKDRAW_LABELS_FNAME)))!
    ).inBatches(of: batchSize).lazy.map {
      makeQuickDrawBatch(samples: $0, flattening: flattening, normalizing: normalizing, 
                     device: device)
    }
  }
}

extension QuickDraw: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`.
  public init(batchSize: Int, on device: Device = Device.default) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device)
  }
}
