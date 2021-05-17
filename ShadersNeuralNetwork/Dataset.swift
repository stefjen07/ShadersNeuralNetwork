//
//  Dataset.swift
//  GPUNN
//
//  Created by Евгений on 16.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

struct DataSample: Codable {
    var image: CGImage
    var texture: MTLTexture
    var label: Int
    
    private enum CodingKeys: String, CodingKey {
        case image
        case label
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(label, forKey: .label)
        try container.encode(image.png!.base64EncodedString(options: []), forKey: .image)
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        label = try container.decode(Int.self, forKey: .label)
        let base64Encoded = try container.decode(String.self, forKey: .image)
        let data = NSData(base64Encoded: base64Encoded, options: [])!
        
        self.image = CGImage(pngDataProviderSource: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        self.texture = image.texture
    }
    
    init(device: MTLDevice, image: CGImage, label: Int) {
        self.image = image
        self.texture = image.texture
        self.label = label
    }
}

struct Dataset: Codable {
    var samples = [DataSample]()
    var classLabels = [String]()
    var imageSize: CGSize = .zero
    
    mutating func updateImageSize() {
        if let sample = samples.first {
            imageSize = CGSize(width: sample.image.width, height: sample.image.height)
        }
    }
    
    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
    
    init(from url: URL) throws {
        let decoder = JSONDecoder()
        let data = try Data(contentsOf: url)
        let set = try decoder.decode(Dataset.self, from: data)
        samples = set.samples
        classLabels = set.classLabels
        imageSize = set.imageSize
    }
    
    init() {
        
    }
    
    init(device: MTLDevice, folderUrl: URL, imageSize: CGSize) {
        let fm = FileManager.default
        do {
            let contents = try fm.contentsOfDirectory(atPath: folderUrl.path)
            var currentLabel = 0
            for subfolder in contents {
                let subfolderPath = folderUrl.appendingPathComponent(subfolder).path
                var isDirectory: ObjCBool = .init(false)
                if fm.fileExists(atPath: subfolderPath, isDirectory: &isDirectory) {
                    if !isDirectory.boolValue {
                        continue
                    }
                    classLabels.append(subfolder)
                    let subfolderContents = try fm.contentsOfDirectory(atPath: subfolderPath)
                    for file in subfolderContents {
                        let fileUrl = URL(fileURLWithPath: subfolderPath).appendingPathComponent(file)
                        if let cgImage = CIImage(contentsOf: fileUrl)?.resize(targetSize: imageSize).inverted.convertedCGImage?.grayscale {
                            let sample = DataSample(device: device, image: cgImage, label: currentLabel)
                            samples.append(sample)
                        }
                    }
                    currentLabel += 1
                }
            }
        } catch {
            print("Unable to get source images.")
        }
    }
    
    func breakInto(trainSet: UnsafeMutablePointer<Dataset>, evaluationSet: UnsafeMutablePointer<Dataset>, evaluationPart: Float) {
        let shuffled = samples.shuffled()
        let testSize = Int(Float(shuffled.count) * evaluationPart)
        trainSet.pointee.classLabels = classLabels
        trainSet.pointee.imageSize = imageSize
        trainSet.pointee.samples = []
        evaluationSet.pointee.classLabels = classLabels
        evaluationSet.pointee.imageSize = imageSize
        evaluationSet.pointee.samples = []
        for i in 0..<testSize {
            evaluationSet.pointee.samples.append(shuffled[i])
        }
        for i in testSize..<shuffled.count {
            trainSet.pointee.samples.append(shuffled[i])
        }
    }
    
    mutating func shuffle() {
        samples.shuffle()
    }
    
    func getTrainingBatch(device: MTLDevice, iteration: Int, batchSize: Int, lossStateBatch: UnsafeMutablePointer<[MPSCNNLossLabels]>) -> [MPSImage] {
        var batch = [MPSImage]()
        for i in 0..<batchSize {
            let idx = iteration * batchSize + i
            let sample = samples[idx]
            
            let image = MPSImage(texture: sample.texture, featureChannels: 1)
            image.label = "trainImage\(i)"
            batch.append(image)
            
            let labelsCount = ((classLabels.count - 1)/4+1)*4
            
            var labelsBuffer = Array(repeating: Float32.zero, count: labelsCount)
            labelsBuffer[sample.label] = Float32(1.0)
            
            
            let labelsData = Data(bytes: labelsBuffer, count: labelsCount * MemoryLayout<Float32>.size)
            guard let labelsDescriptor = MPSCNNLossDataDescriptor(
                data: labelsData,
                layout: .HeightxWidthxFeatureChannels,
                size: .init(width: 1, height: 1, depth: labelsCount)
            ) else {
                print("Unable to create labels descriptor.")
                return batch
            }
            let lossState = MPSCNNLossLabels(device: device, labelsDescriptor: labelsDescriptor)
            lossStateBatch.pointee.append(lossState)
        }
        return batch
    }
}
