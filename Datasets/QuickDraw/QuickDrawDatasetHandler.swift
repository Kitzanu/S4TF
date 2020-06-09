//Datasets/QuickDraw/QuickDrawDatasetHandler.swift  

import Foundation
import TensorFlow

func fetchQuickDrawDataset(
  localStorageDirectory: URL,
  remoteBaseDirectory: String,
  imagesFilename: String,
  labelsFilename: String
) -> [(data: [UInt8], label: Int32)] {
  guard let remoteRoot = URL(string: remoteBaseDirectory) else {
    fatalError("Failed to create QuickDraw root url: \(remoteBaseDirectory)")
  }

  let imagesData = DatasetUtilities.fetchResource(
    filename: imagesFilename,
    fileExtension: "npz",
    remoteRoot: remoteRoot,
    localStorageDirectory: localStorageDirectory)
  let labelsData = DatasetUtilities.fetchResource(
    filename: labelsFilename,
    fileExtension: "npz",
    remoteRoot: remoteRoot,
    localStorageDirectory: localStorageDirectory)

  let images = [UInt8](imagesData).dropFirst(16)
  let labels = [UInt8](labelsData).dropFirst(8).map(Int32.init)
    
  var labeledImages: [(data: [UInt8], label: Int32)] = []

  let imageByteSize = 28 * 28
  for imageIndex in 0..<labels.count {
    let baseAddress = images.startIndex + imageIndex * imageByteSize
    let data = [UInt8](images[baseAddress..<(baseAddress + imageByteSize)])
    labeledImages.append((data: data, label: labels[imageIndex]))
  }

  return labeledImages
}
    
func makeQuickDrawBatch<BatchSamples: Collection>(
  samples: BatchSamples, flattening: Bool, normalizing: Bool, device:Device
) -> LabeledImage where BatchSamples.Element == (data: [UInt8], label: Int32) {
  let bytes = samples.lazy.map(\.data).reduce(into: [], +=)
  let shape: TensorShape = flattening ? [samples.count, 28 * 28] : [samples.count, 28, 28, 1]
  let images = Tensor<UInt8>(shape: shape, scalars: bytes, on:device)
  
  var imageTensor = Tensor<Float>(images) / 255.0
  if normalizing {
    imageTensor = imageTensor * 2.0 - 1.0
  }
  
  let labels = Tensor<Int32>(samples.map(\.label), on: device)
  return LabeledImage(data: imageTensor, label: labels)
}
