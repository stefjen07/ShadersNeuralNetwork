//
//  Dataset.swift
//  GPUNN
//
//  Created by Евгений on 16.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

enum DataType: Int, Codable {
    case image = 0
    case bytes
}

class DataSample: Codable {
    var image: CGImage?
    var texture: MTLTexture?
    var bytes: [UInt8]?
    var type: DataType
    var label: Int
    
    func getMPSImage(device: MTLDevice) -> MPSImage {
        if type == .image {
            return MPSImage(texture: texture!, featureChannels: 1)
        } else {
            let bytes = bytes!
            let descriptor = MPSImageDescriptor(channelFormat: .unorm8, width: 1, height: 1, featureChannels: bytes.count, numberOfImages: 1, usage: [.shaderRead, .shaderWrite])
            let image = MPSImage(device: device, imageDescriptor: descriptor)
            image.writeBytes(bytes, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            return image
        }
    }
    
    private enum CodingKeys: String, CodingKey {
        case image
        case label
        case type
        case bytes
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(label, forKey: .label)
        try container.encode(type, forKey: .type)
        if type == .image {
            try container.encode(image?.png!.base64EncodedString(options: []), forKey: .image)
        } else {
            try container.encode(bytes!, forKey: .bytes)
        }
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        label = try container.decode(Int.self, forKey: .label)
        type = try container.decode(DataType.self, forKey: .type)
        
        if type == .image {
            let base64Encoded = try container.decode(String.self, forKey: .image)
            let data = NSData(base64Encoded: base64Encoded, options: [])!
            
            self.image = CGImage(pngDataProviderSource: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            self.texture = image?.texture
        } else {
            self.bytes = try container.decode([UInt8].self, forKey: .bytes)
        }
    }
    
    init(image: CGImage, label: Int) {
        self.type = .image
        self.label = label
        self.image = image
        self.texture = image.texture
    }
    
    init(bytes: [UInt8], label: Int) {
        self.type = .bytes
        self.bytes = bytes
        self.label = label
    }
}

struct Dataset: Codable {
    var samples = [DataSample]()
    var classLabels = [String]()
    var imageSize: CGSize?
    var type: DataType
    
    private enum CodingKeys: String, CodingKey {
        case samples
        case classLabels
        case imageSize
        case type
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(samples, forKey: .samples)
        try container.encode(classLabels, forKey: .classLabels)
        try container.encode(type, forKey: .type)
        if type == .image {
            try container.encode(imageSize, forKey: .imageSize)
        }
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        samples = try container.decode([DataSample].self, forKey: .samples)
        classLabels = try container.decode([String].self, forKey: .classLabels)
        type = try container.decode(DataType.self, forKey: .type)
        if type == .image {
            do {
                imageSize = try container.decode(CGSize.self, forKey: .imageSize)
            } catch {
                
            }
        }
    }
    
    mutating func optimize() {
        let samplesByClasses = Dictionary(grouping: samples, by: {
            $0.label
        })
        var averageSize = 0
        for i in 0..<classLabels.count {
            if let count = samplesByClasses[i]?.count {
                averageSize += count
            }
        }
        averageSize /= classLabels.count
        var newSamples = [DataSample]()
        for i in 0..<classLabels.count {
            if let classSamples = samplesByClasses[i] {
                newSamples.append(contentsOf: classSamples.dropFirst(averageSize))
            }
        }
        samples = newSamples
    }
    
    mutating func updateImageSize() {
        if let sample = samples.first {
            imageSize = CGSize(width: sample.image!.width, height: sample.image!.height)
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
        type = set.type
    }
    
    init() {
        type = .bytes
    }
    
    init(bytes: [[UInt8]], labels: [Int], classLabels: [String]) {
        for i in bytes.indices {
            samples.append(.init(bytes: bytes[i], label: labels[i]))
        }
        type = .bytes
        self.classLabels = classLabels
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
                        if let cgImage = CIImage(contentsOf: fileUrl)?.inverted.convertedCGImage {
                            var data = [UInt8].init(repeating: 0, count: Int(imageSize.width * imageSize.height))
                            let context = CGContext(data: &data,
                                                            width: Int(imageSize.width),
                                                            height: Int(imageSize.height),
                                                            bitsPerComponent: 8,
                                                            bytesPerRow: Int(imageSize.width),
                                                            space: .init(name: CGColorSpace.linearGray)!,
                                                            bitmapInfo: CGImageAlphaInfo.none.rawValue)!
                            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: imageSize.width, height: imageSize.height))
                            let ciImage = CIImage(bitmapData: Data(bytes: &data, count: Int(imageSize.width * imageSize.height)), bytesPerRow: Int(imageSize.width), size: imageSize, format: .R8, colorSpace: .init(name: CGColorSpace.linearGray)!)
                            let sample = DataSample(image: (ciImage.convertedCGImage?.grayscale(size: imageSize))!, label: currentLabel)
                            samples.append(sample)
                        }
                    }
                    currentLabel += 1
                }
            }
        } catch {
            print("Unable to get source images.")
        }
        type = .image
    }
    
    func breakInto(trainSet: UnsafeMutablePointer<Dataset>, evaluationSet: UnsafeMutablePointer<Dataset>, evaluationPart: Float) {
        let shuffled = samples.shuffled()
        trainSet.pointee.classLabels = classLabels
        trainSet.pointee.imageSize = imageSize
        trainSet.pointee.samples = []
        trainSet.pointee.type = type
        evaluationSet.pointee.classLabels = classLabels
        evaluationSet.pointee.imageSize = imageSize
        evaluationSet.pointee.samples = []
        evaluationSet.pointee.type = type
        let samplesByClasses = Dictionary(grouping: shuffled, by: {
            $0.label
        })
        for i in 0..<classLabels.count {
            guard let classSamples = samplesByClasses[i] else {
                print("Class \(classLabels[i]) samples not found")
                continue
            }
            let evaluationCount = Int(evaluationPart * Float(classSamples.count))
            for j in 0..<classSamples.count {
                if(j<evaluationCount) {
                    evaluationSet.pointee.samples.append(classSamples[j])
                } else {
                    trainSet.pointee.samples.append(classSamples[j])
                }
            }
        }
        return
    }
    
    mutating func shuffle() {
        samples.shuffle()
    }
    
    func getTrainingBatch(device: MTLDevice, iteration: Int, batchSize: Int, lossStateBatch: UnsafeMutablePointer<[MPSCNNLossLabels]>) -> [MPSImage] {
        var batch = [MPSImage]()
        for i in 0..<batchSize {
            let idx = iteration * batchSize + i
            let sample = samples[idx]
            let image = sample.getMPSImage(device: device)
            image.label = "trainImage\(i)"
            batch.append(image)
            
            let labelsCount = classLabels.count
            
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
