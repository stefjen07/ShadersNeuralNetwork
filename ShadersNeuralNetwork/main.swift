//
//  main.swift
//  GPUNN
//
//  Created by Евгений on 11.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

let learningRate: Float = 0.05

autoreleasepool(invoking: {
    if let device = MTLCreateSystemDefaultDevice() {
        if let commandQueue = device.makeCommandQueue() {
            var dataset = getDS()
            let sampleImage = CIImage(mtlTexture: dataset.samples[0].texture, options: [:])!
            sampleImage.saveJPEG("hi.jpg")
            dataset.updateImageSize()
            var trainSet = Dataset(), testSet = Dataset()
            dataset.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2) //have to keep training set full for every class
            
            let filterConst = 32
            
            let layers: [Layer] = [
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 1, outputFC: filterConst, stride: 1, learningRate: learningRate),
                ReLU(),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: filterConst, outputFC: filterConst, stride: 1, learningRate: learningRate),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2),
                Dropout(keepProbability: 0.5),
                
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: filterConst, outputFC: 2*filterConst, stride: 1, learningRate: learningRate),
                ReLU(),
                Convolution(device: device, commandQueue: commandQueue, kernelSize: .init(width: 3, height: 3), inputFC: 2*filterConst, outputFC: 2*filterConst, stride: 1, learningRate: learningRate),
                ReLU(),
                Pooling(mode: .max, filterSize: 2, stride: 2),
                Dropout(keepProbability: 0.5),
                
                Flatten(width: 1600),
                
                Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 1600, outputFC: 256, stride: 1, learningRate: learningRate),
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

extension CIImage {

    func saveJPEG(_ name:String, inDirectoryURL:URL? = nil, quality:CGFloat = 1.0) {
        var destinationURL = inDirectoryURL
        
        if destinationURL == nil {
            destinationURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }
        
        if var destinationURL = destinationURL {
            destinationURL = destinationURL.appendingPathComponent(name)
            if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) {
                do {
                    let context = CIContext()
                    try context.writeJPEGRepresentation(of: self, to: destinationURL, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption : quality])
                } catch {
                    
                }
            }
        }
    }
}
