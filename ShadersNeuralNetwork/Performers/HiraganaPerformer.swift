//
//  HiraganaPerformer.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import CoreImage
import MetalPerformanceShaders

class HiraganaPerformer: BasePerformer {
    override var trainDatasetURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("hiraganaTrain.ds")
    }
    
    override var testDatasetURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("hiraganaTest.ds")
    }
    
    override var modelURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("hiragana.nnm")
    }
    
    override func perform() {
        autoreleasepool(invoking: {
            if !isDatasetPrepared {
                cook(device: device)
                let dataset = getDS()
                var trainSet = Dataset(), testSet = Dataset()
                dataset.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2)
                do {
                    try trainSet.save(to: trainDatasetURL)
                    try testSet.save(to: testDatasetURL)
                } catch {
                    print(error.localizedDescription)
                }
            }
            
            let trainSet = try? Dataset(from: trainDatasetURL)
            let testSet = try? Dataset(from: testDatasetURL)
            guard let trainSet, let testSet else {
                print("Unable to read dataset")
                return
            }
            
            let padding = Padding.valid
            
            let layers: [Layer] = [
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 1, outputFC: 32, stride: 1, learningRate: learningRate, padding: padding),
                ReLU(),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 32, outputFC: 32, stride: 1, learningRate: learningRate, padding: padding),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2, padding: padding),
                //Dropout(keepProbability: 0.1),
                
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 32, outputFC: 64, stride: 1, learningRate: learningRate, padding: padding),
                ReLU(),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 64, outputFC: 64, stride: 1, learningRate: learningRate, padding: padding),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2, padding: padding),
                //Dropout(keepProbability: 0.1),
                
                Flatten(width: 1600),
                
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 1600, outputFC: 256, stride: 1, learningRate: learningRate),
                ReLU(),
                //Dropout(keepProbability: 0.1),
                
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 256, outputFC: 71, stride: 1, learningRate: learningRate)
            ]
            let sampleImage = CIImage(mtlTexture: trainSet.samples[0].texture!, options: [:])!
            sampleImage.saveJPEG("hi.jpg")
            var network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: layers, epochs: 1, batchSize: 128, numberOfClasses: 71)
            if modelExists {
                do {
                    network = try NeuralNetwork(from: modelURL)
                } catch {
                    
                }
            }
            network.train(trainSet: trainSet, evaluationSet: testSet)
            try? network.save(to: modelURL)
        })
    }
}
