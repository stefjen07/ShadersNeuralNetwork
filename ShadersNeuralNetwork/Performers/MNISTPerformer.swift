//
//  MNISTPerformer.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

class MNISTPerformer: BasePerformer {
    override var trainDatasetURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("train.ds")
    }
    
    override var testDatasetURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("test.ds")
    }
    
    override var modelURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("mnist.nnm")
    }
    
    override func perform() {
        autoreleasepool(invoking: {
            if !isDatasetPrepared {
                let trainSet = MNISTDataset(isTrain: true), testSet = MNISTDataset(isTrain: false)
                
                do {
                    try trainSet.load()
                    try testSet.load()
                    trainSet.fillSet()
                    testSet.fillSet()
                    trainSet.set.updateImageSize()
                    testSet.set.updateImageSize()
                    
                    for i in 0..<10 {
                        trainSet.set.classLabels.append(String(i))
                        testSet.set.classLabels.append(String(i))
                    }
                    try trainSet.set.save(to: trainDatasetURL)
                    try testSet.set.save(to: testDatasetURL)
                } catch {
                    print(error.localizedDescription)
                }
            }
            
            let trainDataset = try? Dataset(from: trainDatasetURL)
            let testDataset = try? Dataset(from: testDatasetURL)
            
            guard let trainDataset, let testDataset else {
                print("Unable to read dataset")
                return
            }
            
            let layers: [Layer] = [
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 5, height: 5), inputFC: 1, outputFC: 32, stride: 1, learningRate: learningRate, padding: .same),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2, padding: .same),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 5, height: 5), inputFC: 32, outputFC: 64, stride: 1, learningRate: learningRate, padding: .same),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2, padding: .same),
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 7, height: 7), inputFC: 64, outputFC: 1024, stride: 1, learningRate: learningRate),
                ReLU(),
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 1024, outputFC: 10, stride: 1, learningRate: learningRate)
            ]
            
            let sampleImage = CIImage(mtlTexture: trainDataset.samples[0].texture!, options: [:])!
            sampleImage.saveJPEG("hi.jpg")
            var network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: layers, epochs: 1, batchSize: 40, numberOfClasses: 10)
            
            if modelExists {
                do {
                    network = try NeuralNetwork(from: modelURL)
                } catch {
                    print(error.localizedDescription)
                }
            }
            
            network.train(trainSet: trainDataset, evaluationSet: testDataset)
            try? network.save(to: modelURL)
        })
    }
}
