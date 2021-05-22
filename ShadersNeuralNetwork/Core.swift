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

struct NeuralNetwork: Codable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var epochs: Int
    var batchSize: Int
    var numberOfClasses: Int
    private let doubleBufferingSemaphore = DispatchSemaphore(value: 2)
    private let trainingGraph: MPSNNGraph
    private let inferenceGraph: MPSNNGraph
    private let layers: [Layer]
    
    private enum CodingKeys: String, CodingKey {
        case epochs
        case batchSize
        case layers
        case numberOfClasses
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        let wrappers = layers.map { LayerWrapper($0) }
        try container.encode(wrappers, forKey: .layers)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encode(epochs, forKey: .epochs)
        try container.encode(numberOfClasses, forKey: .numberOfClasses)
    }
    
    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
    
    init(from url: URL) throws {
        let decoder = JSONDecoder()
        let data = try Data(contentsOf: url)
        let model = try decoder.decode(NeuralNetwork.self, from: data)
        self.batchSize = model.batchSize
        self.commandQueue = model.commandQueue
        self.device = model.device
        self.epochs = model.epochs
        self.inferenceGraph = model.inferenceGraph
        self.layers = model.layers
        self.trainingGraph = model.trainingGraph
        self.numberOfClasses = model.numberOfClasses
    }
    
    init(from decoder: Decoder) throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let wrappers = try container.decode([LayerWrapper].self, forKey: .layers)
        let layers = wrappers.map { $0.layer }
        let epochs = try container.decode(Int.self, forKey: .epochs)
        let batchSize = try container.decode(Int.self, forKey: .batchSize)
        let numberOfClasses = try container.decode(Int.self, forKey: .numberOfClasses)
        self.init(device: device, commandQueue: commandQueue, layers: layers, epochs: epochs, batchSize: batchSize, numberOfClasses: numberOfClasses)
    }
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, layers: [Layer], epochs: Int, batchSize: Int, numberOfClasses: Int) {
        self.device = device
        self.commandQueue = commandQueue
        self.epochs = epochs
        self.batchSize = batchSize
        self.layers = layers
        self.numberOfClasses = numberOfClasses
        
        guard let finalNode = createNodes(layers: layers, isTraining: true, batchSize: batchSize, numberOfClasses: numberOfClasses) else {
            fatalError("Unable to get final node of model.")
        }
        
        guard let lossExitPoints = finalNode.trainingGraph(withSourceGradient: nil, nodeHandler: { gradientNode, inferenceNode, inferenceSource, gradientSource in
            gradientNode.resultImage.format = .float32
        }) else {
            fatalError("Unable to get loss exit points.")
        }
        
        assert(lossExitPoints.count == 1)
        
        if let graph = MPSNNGraph(device: device, resultImage: lossExitPoints[0].resultImage, resultImageIsNeeded: true) {
            graph.format = .float32
            trainingGraph = graph
        } else {
            fatalError("Unable to get training graph.")
        }
        
        guard let finalNode = createNodes(layers: layers, isTraining: false, batchSize: batchSize, numberOfClasses: numberOfClasses) else {
            fatalError("Unable to get final node of model.")
        }
        guard let graph = MPSNNGraph(device: device, resultImage: finalNode.resultImage, resultImageIsNeeded: true) else {
            fatalError()
        }
        inferenceGraph = graph
        inferenceGraph.format = .float32
    }
    
    func lossReduceSumAcrossBatch(batch: [MPSImage]) -> Float {
        var ret = Float.zero
        for i in 0..<batch.count {
            var val = [Float.zero]
            batch[i].readBytes(&val, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            ret += Float(val[0]) / Float(batch.count)
        }
        return ret
    }
    
    func getOutputSize(dataset: Dataset) {
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            var lossStateBatch: [MPSCNNLossLabels] = []
            let inputBatch = dataset.getTrainingBatch(device: device, iteration: 0, batchSize: 1, lossStateBatch: &lossStateBatch)
            let outputBatch = encodeInferenceBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: inputBatch)
            
            MPSImageBatchSynchronize(outputBatch, commandBuffer)
            
            let nsOutput = NSArray(array: outputBatch)
            
            commandBuffer.addCompletedHandler() { _ in
                nsOutput.enumerateObjects() { outputImage, idx, stop in
                    if let outputImage = outputImage as? MPSImage {
                        print("Output size: width - \(outputImage.width), height - \(outputImage.height), featureChannels - \(outputImage.featureChannels)")
                    }
                }
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
    
    func encodeTrainingBatchToCommandBuffer(commandBuffer: MTLCommandBuffer, sourceImages: [MPSImage], lossStates: [MPSCNNLossLabels]) -> [MPSImage] {
        
        guard let returnImage = trainingGraph.encodeBatch(to: commandBuffer, sourceImages: [sourceImages], sourceStates: [lossStates], intermediateImages: nil, destinationStates: nil) else {
            print("Unable to encode training batch to command buffer.")
            return []
        }

        MPSImageBatchSynchronize(returnImage, commandBuffer)

        return returnImage
    }
    
    
    func encodeInferenceBatchToCommandBuffer(commandBuffer: MTLCommandBuffer, sourceImages: [MPSImage]) -> [MPSImage] {
        guard let returnImage = inferenceGraph.encodeBatch(to: commandBuffer, sourceImages: [sourceImages], sourceStates: nil, intermediateImages: nil, destinationStates: nil) else {
            print("Unable to encode inference batch to command buffer.")
            return []
        }
        
        return returnImage
    }
    
    private func trainIteration(iteration: Int, numberOfIterations: Int, dataset: Dataset) -> MTLCommandBuffer {
        autoreleasepool(invoking: {
            doubleBufferingSemaphore.wait()
            
            var lossStateBatch: [MPSCNNLossLabels] = []
            
            let commandBuffer = MPSCommandBuffer(from: commandQueue)
            
            let randomTrainBatch = dataset.getTrainingBatch(device: device, iteration: iteration, batchSize: batchSize, lossStateBatch: &lossStateBatch)
            
            let returnBatch = encodeTrainingBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: randomTrainBatch, lossStates: lossStateBatch)
            
            var outputBatch = [MPSImage]()
            
            for i in 0..<batchSize {
                outputBatch.append(lossStateBatch[i].lossImage())
            }
            
            commandBuffer.addCompletedHandler() { commandBuffer in
                doubleBufferingSemaphore.signal()
                
                let trainingLoss = lossReduceSumAcrossBatch(batch: outputBatch)
                print(" Iteration \(iteration+1)/\(numberOfIterations), Training loss = \(trainingLoss)\r", terminator: "")
                fflush(stdout)
                
                let err = commandBuffer.error
                if err != nil {
                    print(err!.localizedDescription)
                }
            }
            
            MPSImageBatchSynchronize(returnBatch, commandBuffer);
            MPSImageBatchSynchronize(outputBatch, commandBuffer);
            
            commandBuffer.commit()
            
            return commandBuffer
        })
    }
    
    private func trainEpoch(dataset: Dataset) -> MTLCommandBuffer {
        let iterations = dataset.samples.count / batchSize
        var latestCommandBuffer: MTLCommandBuffer?
        for i in 0..<iterations {
            latestCommandBuffer = trainIteration(iteration: i, numberOfIterations: iterations, dataset: dataset)
        }
        return latestCommandBuffer!
    }
    
    func predict(samples: [DataSample]) -> [Int] {
        var inputBatch = [MPSImage]()
        for sample in samples {
            inputBatch.append(sample.getMPSImage(device: device))
        }
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
        
        let outputBatch = encodeInferenceBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: inputBatch)
        
        MPSImageBatchSynchronize(outputBatch, commandBuffer)
        
        var anses = [Int]()
        
        commandBuffer.addCompletedHandler() { _ in
            doubleBufferingSemaphore.signal()
            for item in outputBatch.enumerated() {
                let image = item.element
                let size = image.width * image.height * image.featureChannels
                
                var vals = Array(repeating: Float32(-22), count: size)
                var index = -1, maxV = Float32(-100)
                
                image.readBytes(&vals, dataLayout: .featureChannelsxHeightxWidth, imageIndex: 0)
                
                for i in 0..<image.featureChannels {
                    for j in 0..<image.height {
                        for k in 0..<image.width {
                            let val = vals[(i*image.height + j) * image.width + k]
                            if val > maxV {
                                maxV = val
                                index = (i*image.height + j) * image.width + k
                            }
                        }
                    }
                }
                anses.append(index)
            }
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return anses
    }

    func evaluate(dataset: Dataset) {
        autoreleasepool(invoking: {
            var gCorrect = 0
            var gDone = 0
            
            inferenceGraph.reloadFromDataSources()
            
            var imageIdx = 0
            while imageIdx < dataset.samples.count {
                autoreleasepool(invoking: {
                    doubleBufferingSemaphore.wait()
                    
                    var inputBatch = [DataSample]()
                    var labels = [Int]()
                    for i in 0..<min(batchSize, dataset.samples.count-imageIdx) {
                        inputBatch.append(dataset.samples[i+imageIdx])
                        labels.append(dataset.samples[i+imageIdx].label)
                    }
                    
                    let predictions = predict(samples: inputBatch)
                    
                    for i in 0..<predictions.count {
                        //print("\(labels[i]) \(predictions[i])")
                        if labels[i] == predictions[i] {
                            gCorrect += 1
                        }
                        gDone += 1
                    }
                    
                    imageIdx += batchSize
                })
            }
            
            print("Test Set Accuracy = \(Float(gCorrect) / Float(gDone) * 100.0) %")
        })
    }

    func train(trainSet: Dataset, evaluationSet: Dataset) {
        // Use double buffering to keep the gpu completely busy.
        evaluate(dataset: evaluationSet)
        for i in 0..<epochs {
            autoreleasepool(invoking: {
                print("Starting epoch \(i)")
                trainEpoch(dataset: trainSet).waitUntilCompleted()
                evaluate(dataset: evaluationSet)
            })
        }
        evaluate(dataset: evaluationSet)
    }
    
    func hi() {
        print("Hi!")
    }
}

extension CGImage {
    
    var texture: MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let texture = device.makeTexture(descriptor: descriptor)!
        let region = MTLRegion(origin: .init(x: 0, y: 0, z: 0), size: .init(width: width, height: height, depth: 1))
        
        var format = vImage_CGImageFormat(bitsPerComponent: UInt32(8), bitsPerPixel: UInt32(8), colorSpace: Unmanaged.passRetained(CGColorSpace(name: CGColorSpace.linearGray)!), bitmapInfo: .init(rawValue: CGImageAlphaInfo.none.rawValue), version: 0, decode: nil, renderingIntent: .defaultIntent)
        do {
            var sourceBuffer = try vImage_Buffer(cgImage: self, format: format)
            var error = vImage_Error()
            let destImage = vImageCreateCGImageFromBuffer(&sourceBuffer, &format, nil, nil, numericCast(kvImageNoFlags), &error).takeRetainedValue()
            
            guard error == noErr else {
                fatalError()
            }
            
            let dstData = destImage.dataProvider?.data
            let pixelData = CFDataGetBytePtr(dstData!)
            
            texture.replace(region: region, mipmapLevel: 0, withBytes: pixelData!, bytesPerRow: bytesPerRow)
            
            return texture
        } catch {
            fatalError(error.localizedDescription)
        }
    }
    
    func grayscale(size: CGSize) -> CGImage {
        let imageRect:CGRect = CGRect(origin: .zero, size: size)
        let colorSpace = CGColorSpace(name: CGColorSpace.linearGray)!
        let width = size.width
        let height = size.height
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let context = CGContext(data: nil, width: Int(width), height: Int(height), bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
        context?.draw(self, in: imageRect)
        if let makeImg = context?.makeImage() {
            return makeImg
        }
        return self
    }
    
    var png: NSData? {
        guard let mutableData = CFDataCreateMutable(nil, 0),
              let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { print("Unable to get PNG data"); return nil }
        CGImageDestinationAddImage(destination, self, nil)
        guard CGImageDestinationFinalize(destination) else { print("Unable to get PNG data"); return nil }
        return mutableData as NSData
    }
}

extension CIImage {
    var convertedCGImage: CGImage? {
        let context = CIContext(options: nil)
        return context.createCGImage(self, from: self.extent)
    }
    
    var inverted: CIImage {
        let filter = CIFilter(name: "CIColorInvert")!
        filter.setValue(self, forKey: kCIInputImageKey)
        
        return filter.outputImage ?? self
    }
    
    func resize(targetSize: CGSize) -> CIImage {
        let resizeFilter = CIFilter(name:"CILanczosScaleTransform")!

        let scale = targetSize.height / self.extent.height
        let aspectRatio = targetSize.width/(self.extent.width * scale)

        resizeFilter.setValue(self, forKey: kCIInputImageKey)
        resizeFilter.setValue(scale, forKey: kCIInputScaleKey)
        resizeFilter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        return resizeFilter.outputImage ?? self
    }
}
