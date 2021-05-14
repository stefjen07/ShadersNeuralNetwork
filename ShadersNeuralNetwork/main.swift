//
//  main.swift
//  GPUNN
//
//  Created by Евгений on 11.05.2021.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

struct DataSample {
    var image: CGImage
    var label: Int
}

struct Dataset {
    var samples = [DataSample]()
    var classLabels = [String]()
    var imageSize: CGSize = .zero
    
    func breakInto(trainSet: UnsafeMutablePointer<Dataset>, evaluationSet: UnsafeMutablePointer<Dataset>, evaluationPart: Double) {
        
    }
    
    func getRandomTrainingBatch(device: MTLDevice, batchSize: Int, lossStateBatch: UnsafeMutablePointer<[MPSCNNLossLabels]>) -> [MPSImage] {
        let descriptor = MPSImageDescriptor(channelFormat: .unorm8, width: Int(imageSize.width), height: Int(imageSize.height), featureChannels: 1, numberOfImages: 1, usage: [.shaderRead, .shaderWrite])
        var batch = [MPSImage]()
        var lossStateBatchOut = [MPSCNNLossLabels]()
        for i in 0..<batchSize {
            let randomVal = Float(arc4random()) / Float(RAND_MAX)
            let randomIdx = Int(randomVal * Float(samples.count))
            let sample = samples[randomIdx]
            
            let image = MPSImage(device: device, imageDescriptor: descriptor)
            image.label = "trainImage\(i)"
            batch.append(image)
            
            guard let data = sample.image.png else {
                print("Unable to get image.")
                return batch
            }
            
            image.writeBytes(data.bytes, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            
            let labelsCount = (classLabels.count / 4 + 1) * 4
            
            var labelsBuffer = Array(repeating: Float.zero, count: labelsCount)
            labelsBuffer[sample.label] = 1
            
            
            let labelsData = Data(bytes: labelsBuffer, count: labelsCount * MemoryLayout<Float>.size)
            guard let labelsDescriptor = MPSCNNLossDataDescriptor(
                data: labelsData,
                layout: .HeightxWidthxFeatureChannels,
                size: .init(width: 1, height: 1, depth: labelsCount)
            ) else {
                print("Unable to create labels descriptor.")
                return batch
            }
            let lossState = MPSCNNLossLabels(device: device, labelsDescriptor: labelsDescriptor)
            lossStateBatchOut.append(lossState)
        }
        lossStateBatch.pointee = lossStateBatchOut
        return batch
    }
}

class Layer {
    
}

class Convolution: Layer {
    let dataSource: ConvDataSource
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, kernelSize: CGSize, inputFC: Int, outputFC: Int, stride: Int, learningRate: Float) {
        dataSource = .init(device: device, kernelWidth: Int(kernelSize.width), kernelHeight: Int(kernelSize.height), inputFeatureChannels: inputFC, outputFeatureChannnels: outputFC, stride: stride, learningRate: learningRate, commandQueue: commandQueue)
    }
}

enum PoolingMode {
    case max
    case average
}

class Pooling: Layer {
    let mode: PoolingMode
    let filterSize: Int
    let stride: Int
    
    init(mode: PoolingMode, filterSize: Int, stride: Int) {
        self.mode = mode
        self.filterSize = filterSize
        self.stride = stride
    }
}

class ReLU: Layer {
    
}

class Sigmoid: Layer {
    
}

class Dropout: Layer {
    let keepProbability: Float
    
    init(keepProbability: Float) {
        self.keepProbability = keepProbability
    }
}

class Dense: Layer {
    let dataSource: ConvDataSource
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, kernelSize: CGSize, inputFC: Int, outputFC: Int, stride: Int, learningRate: Float) {
        dataSource = .init(device: device, kernelWidth: Int(kernelSize.width), kernelHeight: Int(kernelSize.height), inputFeatureChannels: inputFC, outputFeatureChannnels: outputFC, stride: stride, learningRate: learningRate, commandQueue: commandQueue)
    }
}

func createNodes(layers: [Layer], isTraining: Bool, batchSize: Int) -> MPSNNFilterNode? {
    var source = MPSNNImageNode(handle: nil)
    var lastNode: MPSNNFilterNode? = nil
    for layer in layers {
        switch layer {
        case let layer as Convolution:
            lastNode = MPSCNNConvolutionNode(source: source, weights: layer.dataSource.convolution.dataSource)
        case let layer as Pooling:
            if layer.mode == .max {
                lastNode = MPSCNNPoolingMaxNode(source: source, filterSize: layer.filterSize, stride: layer.stride)
            } else {
                lastNode = MPSCNNPoolingAverageNode(source: source, filterSize: layer.filterSize, stride: layer.stride)
            }
        case _ as ReLU:
            lastNode = MPSCNNNeuronReLUNode(source: source)
        case _ as Sigmoid:
            lastNode = MPSCNNNeuronSigmoidNode(source: source)
        case let layer as Dropout:
            lastNode = MPSCNNDropoutNode(source: source, keepProbability: layer.keepProbability)
        case let layer as Dense:
            lastNode = MPSCNNFullyConnectedNode(source: source, weights: layer.dataSource.convolution.dataSource)
        default:
            break
        }
        source = lastNode!.resultImage
    }
    if isTraining {
        let lossDescriptor = MPSCNNLossDescriptor(type: .softMaxCrossEntropy, reductionType: .sum)
        lossDescriptor.weight = 1.0 / Float(batchSize)
        return MPSCNNLossNode(source: source, lossDescriptor: lossDescriptor)
    } else {
        return MPSCNNSoftMaxNode(source: source)
    }
}

struct NeuralNetwork {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var epochs: Int
    var batchSize: Int
    var evaluationFreq: Int
    private let doubleBufferingSemaphore = DispatchSemaphore(value: 2)
    private let trainingGraph: MPSNNGraph
    private let inferenceGraph: MPSNNGraph
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, layers: [Layer], epochs: Int, batchSize: Int, evaluationFreq: Int) {
        self.device = device
        self.commandQueue = commandQueue
        self.epochs = epochs
        self.batchSize = batchSize
        self.evaluationFreq = evaluationFreq
        
        guard let finalNode = createNodes(layers: layers, isTraining: true, batchSize: batchSize) else {
            fatalError("Unable to get final node of model.")
        }
        
        guard let lossExitPoints = finalNode.trainingGraph(withSourceGradient: nil, nodeHandler: { gradientNode, inferenceNode, inferenceSource, gradientSource in
            gradientNode.resultImage.format = .float32
        }) else {
            fatalError("Unable to get loss exit points.")
        }
        
        assert( lossExitPoints.count == 1);
        
        trainingGraph = MPSNNGraph(device: device, resultImage: lossExitPoints[0].resultImage, resultImageIsNeeded: true) ?? MPSNNGraph()
        
        trainingGraph.format = .float16
        
        guard let finalNode = createNodes(layers: layers, isTraining: false, batchSize: batchSize) else {
            fatalError("Unable to get final node of model.")
        }
        inferenceGraph = MPSNNGraph(device: device, resultImage: finalNode.resultImage, resultImageIsNeeded: true) ?? MPSNNGraph()
        inferenceGraph.format = .float16
    }
    
    func lossReduceSumAcrossBatch(batch: [MPSImage]) -> Float {
        var ret = Float.zero
        for i in 0..<batch.count {
            let curr = batch[i]
            var val = Array(repeating: Float.zero, count: 1)
            assert(curr.width * curr.height * curr.featureChannels == 1)
            curr.readBytes(&val, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            ret += val[0] / Float(batchSize)
        }
        return ret
    }
    
    func encodeTrainingBatchToCommandBuffer(commandBuffer: MTLCommandBuffer, sourceImages: [MPSImage], lossStates: [MPSCNNLossLabels]) -> [MPSImage] {
        
        guard let returnImage = trainingGraph.encodeBatch(to: commandBuffer, sourceImages: [sourceImages], sourceStates: [lossStates]) else {
            return []
        }

        MPSImageBatchSynchronize(returnImage, commandBuffer)

        return returnImage
    }
    
    
    func encodeInferenceBatchToCommandBuffer(commandBuffer: MTLCommandBuffer, sourceImages: [MPSImage]) -> [MPSImage] {
        return inferenceGraph.encodeBatch(to: commandBuffer, sourceImages: [sourceImages], sourceStates: nil) ?? []
    }
    
    private func trainIteration(dataset: Dataset) -> MTLCommandBuffer {
        doubleBufferingSemaphore.wait()
        var lossStateBatch: [MPSCNNLossLabels] = []
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Unable to make command buffer.")
        }
        
        let randomTrainBatch = dataset.getRandomTrainingBatch(device: device, batchSize: batchSize, lossStateBatch: &lossStateBatch)
        
        let returnBatch = encodeTrainingBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: randomTrainBatch, lossStates: lossStateBatch)
        
        var outputBatch = [MPSImage]()
        
        for i in 0..<batchSize {
            outputBatch.append(lossStateBatch[i].lossImage())
        }
        
        var iteration = 1
        
        commandBuffer.addCompletedHandler() { commandBuffer in
            doubleBufferingSemaphore.signal()
            
            let trainingLoss = lossReduceSumAcrossBatch(batch: outputBatch)
            print("Iteration \(iteration), Training loss = \(trainingLoss)")
            iteration += 1
            
            let err = commandBuffer.error
            if err != nil {
                print(err!.localizedDescription)
            }
        }
        
        MPSImageBatchSynchronize(returnBatch, commandBuffer);
        MPSImageBatchSynchronize(outputBatch, commandBuffer);
        
        commandBuffer.commit()
        
        return commandBuffer
    }
    
    func checkLabel(image: MPSImage, label: Int) -> Bool {
        assert(image.numberOfImages == 1)
        
        let size = image.width * image.height * image.featureChannels
        
        var vals = Array(repeating: Float(-22), count: size)
        
        var index = -1, maxV = Float(-100)
        
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
        
        return index == label
    }
    
    func evaluate(dataset: Dataset) {
        var gCorrect = 0
        
        inferenceGraph.reloadFromDataSources()
        
        var inputDescriptor = MPSImageDescriptor(channelFormat: .unorm8, width: Int(dataset.imageSize.width), height: Int(dataset.imageSize.height), featureChannels: 1, numberOfImages: 1, usage: .shaderRead)
        
        var latestCommandBuffer: MPSCommandBuffer? = nil
        
        var imageIdx = 0
        while imageIdx < dataset.samples.count {
            doubleBufferingSemaphore.wait()
            
            let sample = dataset.samples[imageIdx]
            
            var inputBatch = [MPSImage]()
            for i in 0..<batchSize {
                let inputImage = MPSImage(device: device, imageDescriptor: inputDescriptor)
                inputBatch.append(inputImage)
            }
            
            var commandBuffer = MPSCommandBuffer(from: commandQueue)
            
            let nsInput = NSArray(array: inputBatch)
            
            guard var data = sample.image.png else {
                print("Unable to get image.")
                continue
            }
            
            nsInput.enumerateObjects() { inputImage, idx, stop in
                if let inputImage = inputImage as? MPSImage {
                    inputImage.writeBytes(&data, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
                }
            }
            
            inputBatch = Array(_immutableCocoaArray: nsInput)
            
            var outputBatch = encodeInferenceBatchToCommandBuffer(commandBuffer: commandBuffer, sourceImages: inputBatch)
            
            MPSImageBatchSynchronize(outputBatch, commandBuffer)
            
            let nsOutput = NSArray(array: outputBatch)
            
            commandBuffer.addCompletedHandler() { _ in
                doubleBufferingSemaphore.signal()
                
                nsOutput.enumerateObjects() { outputImage, idx, stop in
                    if let outputImage = outputImage as? MPSImage {
                        if checkLabel(image: outputImage, label: sample.label) {
                            gCorrect += 1
                        }
                    }
                }
            }
            
            commandBuffer.commit()
            latestCommandBuffer = commandBuffer
            
            imageIdx += batchSize
        }
        
        latestCommandBuffer?.waitUntilCompleted()
        print("Test Set Accuracy = \(Float(gCorrect) / (Float(dataset.samples.count) / 100.0 ))")
    }

    func train(trainSet: Dataset, evaluationSet: Dataset) {
        // Use double buffering to keep the gpu completely busy.
        var latestCommandBuffer: MTLCommandBuffer? = nil
        for i in 0..<epochs {
            if i % evaluationFreq == 0 {
                if latestCommandBuffer != nil {
                    latestCommandBuffer?.waitUntilCompleted()
                }
                evaluate(dataset: evaluationSet)
            }
            latestCommandBuffer = trainIteration(dataset: trainSet)
        }
    }
    
    func hi() {
        print("Hi!")
    }
}

if let device = MTLCreateSystemDefaultDevice() {
    if let commandQueue = device.makeCommandQueue() {
        let network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: [], epochs: 30, batchSize: 16, evaluationFreq: 2)
        network.hi()
    } else {
        print("Unable to get command queue.")
    }
} else {
    print("Unable to get GPU device.")
}

extension CGImage {
    var png: NSData? {
        guard let mutableData = CFDataCreateMutable(nil, 0),
            let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, self, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as NSData
    }
}
