//
//  NeuralNetwork.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import MetalPerformanceShaders

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
}
