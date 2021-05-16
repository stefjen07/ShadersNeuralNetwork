//
//  main.swift
//  GPUNN
//
//  Created by Евгений on 11.05.2021.
//

import Foundation
import MetalPerformanceShaders

let learningRate: Float = 0.05

autoreleasepool(invoking: {
    if let device = MTLCreateSystemDefaultDevice() {
        if let commandQueue = device.makeCommandQueue() {
            var dataset = getDS()
            dataset.updateImageSize()
            var trainSet = Dataset(), testSet = Dataset()
            dataset.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2) //have to keep training set full for every class
            
            let layers: [Layer] = [
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 1, outputFC: 32, stride: 1, learningRate: learningRate),
                ReLU(),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 32, outputFC: 32, stride: 1, learningRate: learningRate),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2),
                Dropout(keepProbability: 0.5),
                
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 32, outputFC: 64, stride: 1, learningRate: learningRate),
                ReLU(),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 64, outputFC: 64, stride: 1, learningRate: learningRate),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2),
                Dropout(keepProbability: 0.5),
                
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 8, height: 8), inputFC: 64, outputFC: 256, stride: 1, learningRate: learningRate),
                ReLU(),
                Dropout(keepProbability: 0.5),
                
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 256, outputFC: dataset.classLabels.count, stride: 1, learningRate: learningRate)
            ]
            let network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: layers, epochs: 30, batchSize: 16)
            network.hi()
            network.getOutputSize(dataset: dataset)
            network.train(trainSet: trainSet, evaluationSet: testSet)
        } else {
            print("Unable to get command queue.")
        }
    } else {
        print("Unable to get GPU device.")
    }
})
