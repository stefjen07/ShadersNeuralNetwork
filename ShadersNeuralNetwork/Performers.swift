//
//  Performers.swift
//  GPUNN
//
//  Created by Евгений on 19.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

func performHiragana(learningRate: Float) {
    autoreleasepool(invoking: {
        if let device = MTLCreateSystemDefaultDevice() {
            if let commandQueue = device.makeCommandQueue() {
                
                var dataset = getDS()
                let sampleImage = CIImage(mtlTexture: dataset.samples[0].texture, options: [:])!
                sampleImage.saveJPEG("hi.jpg")
                dataset.updateImageSize()
                var trainSet = Dataset(), testSet = Dataset()
                dataset.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2)
                let padding = Padding.valid
                
                var layers: [Layer] = [
                    Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 1, outputFC: 32, stride: 1, learningRate: learningRate, padding: padding),
                    ReLU(),
                    Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 32, outputFC: 64, stride: 1, learningRate: learningRate, padding: padding),
                    ReLU(),
                    Pooling(mode: .max, filterSize: 2, stride: 2, padding: padding),
                    //Dropout(keepProbability: 0.5),
                    
                    Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 64, outputFC: 64, stride: 1, learningRate: learningRate, padding: padding),
                    ReLU(),
                    Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 64, outputFC: 64, stride: 1, learningRate: learningRate, padding: padding),
                    ReLU(),
                    Pooling(mode: .max, filterSize: 2, stride: 2, padding: padding),
                    //Dropout(keepProbability: 0.5),
                    
                    Flatten(width: 1600),
                    
                    Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 1600, outputFC: 256, stride: 1, learningRate: learningRate),
                    ReLU(),
                    //Dropout(keepProbability: 0.5),
                    
                    Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 256, outputFC: dataset.classLabels.count, stride: 1, learningRate: learningRate)
                ]
                let network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: &layers, epochs: 1, batchSize: 40, numberOfClasses: dataset.classLabels.count)
                sampleImage.saveJPEG("hi.jpg")
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
}

func performMNIST(learningRate: Float) {
    autoreleasepool(invoking: {
        if let device = MTLCreateSystemDefaultDevice() {
            if let commandQueue = device.makeCommandQueue() {
                /*var trainSet = MNISTDataset(isTrain: true), testSet = MNISTDataset(isTrain: false)
                do {
                    try trainSet.load()
                    try testSet.load()
                    trainSet.fillSet(device: device)
                    testSet.fillSet(device: device)
                    try trainSet.set.save(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("train.ds"))
                    try testSet.set.save(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("test.ds"))
                } catch {
                    
                }*/
                let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                let trainSet = try? Dataset(from: url.appendingPathComponent("train.ds"))
                let testSet = try? Dataset(from: url.appendingPathComponent("test.ds"))
                guard var trainSet = trainSet, var testSet = testSet else {
                    return
                }
                
                //trainSet.updateImageSize()
                //testSet.updateImageSize()
                
                //for i in 0..<10 {
                    //trainSet.classLabels.append(String(i))
                    //testSet.classLabels.append(String(i))
                //}
                
                //try? trainSet.save(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("train.ds"))
                //try? testSet.save(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("test.ds"))
                
                
                var layers: [Layer] = [
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
                let sampleImage = CIImage(mtlTexture: trainSet.samples[0].texture, options: [:])!
                sampleImage.saveJPEG("hi.jpg")
                let network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: &layers, epochs: 1, batchSize: 40, numberOfClasses: 10)
                network.hi()
                network.train(trainSet: trainSet, evaluationSet: testSet)
            } else {
                print("Unable to get command queue.")
            }
        } else {
            print("Unable to get GPU device.")
        }
    })
}
