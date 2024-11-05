//
//  Core.swift
//  GPUNN
//
//  Created by Евгений on 16.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage
import Accelerate

var validConvPadding = MPSNNDefaultPadding(method: .validOnly)
var sameConvPadding = MPSNNDefaultPadding(method: .sizeSame)

public struct LayerWrapper: Codable {
    let layer: Layer
    
    private enum CodingKeys: String, CodingKey {
        case base
        case payload
    }

    private enum Base: Int, Codable {
        case dense = 0
        case convolution
        case pooling
        case relu
        case sigmoid
        case dropout
        case flatten
    }
    
    init(_ layer: Layer) {
        self.layer = layer
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch layer {
        case let payload as Dense:
            try container.encode(Base.dense, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Convolution:
            try container.encode(Base.convolution, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Pooling:
            try container.encode(Base.pooling, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as ReLU:
            try container.encode(Base.relu, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Sigmoid:
            try container.encode(Base.sigmoid, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Dropout:
            try container.encode(Base.dropout, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Flatten:
            try container.encode(Base.flatten, forKey: .base)
            try container.encode(payload, forKey: .payload)
        default:
            fatalError()
        }
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let base = try container.decode(Base.self, forKey: .base)
        
        switch base {
        case .dense:
            self.layer = try container.decode(Dense.self, forKey: .payload)
        case .convolution:
            self.layer = try container.decode(Convolution.self, forKey: .payload)
        case .pooling:
            self.layer = try container.decode(Pooling.self, forKey: .payload)
        case .relu:
            self.layer = try container.decode(ReLU.self, forKey: .payload)
        case .sigmoid:
            self.layer = try container.decode(Sigmoid.self, forKey: .payload)
        case .dropout:
            self.layer = try container.decode(Dropout.self, forKey: .payload)
        case .flatten:
            self.layer = try container.decode(Flatten.self, forKey: .payload)
        }
    }

}

class Layer: Codable {
    
}

var num = 0

enum Padding: Int, Codable {
    case same
    case valid
}

enum PoolingMode: Int, Codable {
    case max
    case average
}

class Convolution: Layer {
    let dataSource: ConvDataSource
    let padding: Padding
    
    private enum CodingKeys: String, CodingKey {
        case dataSource
        case padding
    }
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, kernelSize: CGSize, inputFC: Int, outputFC: Int, stride: Int, learningRate: Float, padding: Padding) {
        dataSource = .init(device: device, kernelWidth: Int(kernelSize.width), kernelHeight: Int(kernelSize.height), inputFeatureChannels: inputFC, outputFeatureChannnels: outputFC, stride: stride, learningRate: learningRate, commandQueue: commandQueue, num: num)
        self.padding = padding
        num += 1
        super.init()
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(dataSource, forKey: .dataSource)
        try container.encode(padding, forKey: .padding)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        padding = try container.decode(Padding.self, forKey: .padding)
        dataSource = try container.decode(ConvDataSource.self, forKey: .dataSource)
        try super.init(from: decoder)
    }
}

class Pooling: Layer {
    let mode: PoolingMode
    let filterSize: Int
    let stride: Int
    let padding: Padding
    
    private enum CodingKeys: String, CodingKey {
        case mode
        case filterSize
        case stride
        case padding
    }
    
    init(mode: PoolingMode, filterSize: Int, stride: Int, padding: Padding) {
        self.mode = mode
        self.filterSize = filterSize
        self.stride = stride
        self.padding = padding
        super.init()
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(mode, forKey: .mode)
        try container.encode(filterSize, forKey: .filterSize)
        try container.encode(stride, forKey: .stride)
        try container.encode(padding, forKey: .padding)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        mode = try container.decode(PoolingMode.self, forKey: .mode)
        filterSize = try container.decode(Int.self, forKey: .filterSize)
        stride = try container.decode(Int.self, forKey: .stride)
        padding = try container.decode(Padding.self, forKey: .padding)
        try super.init(from: decoder)
    }
}

class ReLU: Layer {
    
}

class Sigmoid: Layer {
    
}

class Dropout: Layer {
    let keepProbability: Float
    
    private enum CodingKeys: String, CodingKey {
        case keepProbability
    }
    
    init(keepProbability: Float) {
        self.keepProbability = keepProbability
        super.init()
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(keepProbability, forKey: .keepProbability)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        keepProbability = try container.decode(Float.self, forKey: .keepProbability)
        try super.init(from: decoder)
    }
}

class Dense: Layer {
    let dataSource: ConvDataSource
    
    private enum CodingKeys: String, CodingKey {
        case dataSource
    }
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, kernelSize: CGSize, inputFC: Int, outputFC: Int, stride: Int, learningRate: Float) {
        dataSource = .init(device: device, kernelWidth: Int(kernelSize.width), kernelHeight: Int(kernelSize.height), inputFeatureChannels: inputFC, outputFeatureChannnels: outputFC, stride: stride, learningRate: learningRate, commandQueue: commandQueue, num: num)
        num += 1
        super.init()
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(dataSource, forKey: .dataSource)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        dataSource = try container.decode(ConvDataSource.self, forKey: .dataSource)
        try super.init(from: decoder)
    }
}

class Flatten: Layer {
    let width: Int
    
    private enum CodingKeys: String, CodingKey {
        case width
    }
    
    init(width: Int) {
        self.width = width
        super.init()
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(width, forKey: .width)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        width = try container.decode(Int.self, forKey: .width)
        try super.init(from: decoder)
    }
}

func createNodes(layers: [Layer], isTraining: Bool, batchSize: Int, numberOfClasses: Int) -> MPSNNFilterNode? {
    var source = MPSNNImageNode(handle: nil)
    var lastNode: MPSNNFilterNode? = nil
    for layer in layers {
        switch layer {
        case let layer as Convolution:
            let conv = MPSCNNConvolutionNode(source: source, weights: layer.dataSource)
            if layer.padding == .valid {
                conv.paddingPolicy = validConvPadding
            } else {
                conv.paddingPolicy = sameConvPadding
            }
            lastNode = conv
        case let layer as Pooling:
            if layer.mode == .max {
                let node = MPSCNNPoolingMaxNode(source: source, filterSize: layer.filterSize, stride: layer.stride)
                if layer.padding == .valid {
                    node.paddingPolicy = validConvPadding
                } else {
                    node.paddingPolicy = sameConvPadding
                }
                lastNode = node
            } else {
                let node = MPSCNNPoolingAverageNode(source: source, filterSize: layer.filterSize, stride: layer.stride)
                if layer.padding == .valid {
                    node.paddingPolicy = validConvPadding
                } else {
                    node.paddingPolicy = sameConvPadding
                }
                lastNode = node
            }
        case _ as ReLU:
            lastNode = MPSCNNNeuronReLUNode(source: source, a: 0.0)
        case _ as Sigmoid:
            lastNode = MPSCNNNeuronSigmoidNode(source: source)
        case let layer as Flatten:
            lastNode = MPSNNReshapeNode(source: source, resultWidth: 1, resultHeight: 1, resultFeatureChannels: layer.width)
        case let layer as Dropout:
            if isTraining {
                lastNode = MPSCNNDropoutNode(source: source, keepProbability: layer.keepProbability)
            }
        case let layer as Dense:
            lastNode = MPSCNNFullyConnectedNode(source: source, weights: layer.dataSource)
        default:
            break
        }
        source = lastNode!.resultImage
        source.format = .float32
    }
    if isTraining {
        let lossDescriptor = MPSCNNLossDescriptor(type: .softMaxCrossEntropy, reductionType: .mean)
        //lossDescriptor.numberOfClasses = numberOfClasses
        lossDescriptor.weight = 1.0 / Float(batchSize)
        return MPSCNNLossNode(source: source, lossDescriptor: lossDescriptor)
    } else {
        return MPSCNNSoftMaxNode(source: source)
    }
}
