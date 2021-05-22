//
//  Performers.swift
//  GPUNN
//
//  Created by Евгений on 19.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

func performNPL(learningRate: Float, firstTime: Bool) {
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let inputSize = 16
    let text = try? String(contentsOf: url.appendingPathComponent("shakespeare.html"))
    guard let text = text else {
        fatalError()
    }
    var bytesArray = [[UInt8]]()
    var labels = [Int]()
    for i in inputSize-1..<text.count-1 {
        let end = text.index(text.startIndex, offsetBy: i)
        let start = text.index(end, offsetBy: 1-inputSize)
        var bytes = [UInt8].init(repeating: .zero, count: inputSize)
        var length = 0
        var remaining = start..<text.index(after: end)
        text.getBytes(&bytes, maxLength: inputSize, usedLength: &length, encoding: .unicode, range: start...end, remaining: &remaining)
        bytesArray.append(bytes)
        labels.append(Int(text[text.index(after: end)].asciiValue!))
    }
    autoreleasepool(invoking: {
        
        let set = Dataset(bytes: bytesArray, labels: labels, classLabels: (1...256).map { String($0) })
        var trainSet = Dataset(), testSet = Dataset()
        set.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2)
        try? trainSet.save(to: url.appendingPathComponent("textTrain.ds"))
        try? testSet.save(to: url.appendingPathComponent("textTest.ds"))
        //let trainSet = try? Dataset(from: url.appendingPathComponent("textTrain.ds"))
        //let testSet = try? Dataset(from: url.appendingPathComponent("textTest.ds"))
        //guard let trainSet = trainSet, let testSet = testSet else {
        //    fatalError()
        //}
        if let device = MTLCreateSystemDefaultDevice() {
            if let commandQueue = device.makeCommandQueue() {
                var layers: [Layer] = [
                    Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: inputSize, height: 1), inputFC: 1, outputFC: 32, stride: 1, learningRate: learningRate),
                    Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 32, outputFC: 128, stride: 1, learningRate: learningRate),
                    Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 128, outputFC: 1024, stride: 1, learningRate: learningRate),
                    Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 1024, outputFC: 256, stride: 1, learningRate: learningRate)
                ]
                let network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: layers, epochs: 1, batchSize: 32, numberOfClasses: 256)
                network.hi()
                network.train(trainSet: trainSet, evaluationSet: testSet)
                var input = "Shakespeare"
                var res = input
                let additionSize = 20
                for _ in 0..<additionSize {
                    var bytes = [UInt8].init(repeating: .zero, count: inputSize)
                    var length = 0
                    var remaining = input.startIndex..<input.endIndex
                    input.getBytes(&bytes, maxLength: inputSize, usedLength: &length, encoding: .ascii, range: input.startIndex..<input.endIndex, remaining: &remaining)
                    let sample = [
                        DataSample(bytes: bytes, label: -1)
                    ]
                    let newChar = Character(Unicode.Scalar(network.predict(samples: sample).first!)!)
                    input.removeFirst()
                    input.append(newChar)
                    res.append(newChar)
                }
                print(res)
            }
        }
    })
}

func performHiragana(learningRate: Float, firstTime: Bool, fromFile: Bool) {
    autoreleasepool(invoking: {
        if let device = MTLCreateSystemDefaultDevice() {
            if let commandQueue = device.makeCommandQueue() {
                let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                if firstTime {
                    cook(device: device)
                    let dataset = getDS()
                    var trainSet = Dataset(), testSet = Dataset()
                    dataset.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2)
                    do {
                        try trainSet.save(to: url.appendingPathComponent("hiraganaTrain.ds"))
                        try testSet.save(to: url.appendingPathComponent("hiraganaTest.ds"))
                    } catch {
                        fatalError("Error")
                    }
                }
                
                let trainSet = try? Dataset(from: url.appendingPathComponent("hiraganaTrain.ds"))
                let testSet = try? Dataset(from: url.appendingPathComponent("hiraganaTest.ds"))
                guard let trainSet = trainSet, let testSet = testSet else {
                    return
                }
                
                let padding = Padding.valid
                
                var layers: [Layer] = [
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
                if fromFile {
                    network = try! NeuralNetwork(from: url.appendingPathComponent("hiragana.nnm"))
                }
                network.hi()
                //network.getOutputSize(dataset: trainSet)
                network.train(trainSet: trainSet, evaluationSet: testSet)
                try? network.save(to: url.appendingPathComponent("hiragana.nnm"))
            } else {
                print("Unable to get command queue.")
            }
        } else {
            print("Unable to get GPU device.")
        }
    })
}

func performMNIST(learningRate: Float, firstTime: Bool) {
    autoreleasepool(invoking: {
        if let device = MTLCreateSystemDefaultDevice() {
            if let commandQueue = device.makeCommandQueue() {
                if firstTime {
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
                        try trainSet.set.save(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("train.ds"))
                        try testSet.set.save(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("test.ds"))
                    } catch {
                        
                    }
                }
                
                let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                let trainSet = try? Dataset(from: url.appendingPathComponent("train.ds"))
                let testSet = try? Dataset(from: url.appendingPathComponent("test.ds"))
                
                guard let trainSet = trainSet, let testSet = testSet else {
                    return
                }
                
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
                
                let sampleImage = CIImage(mtlTexture: trainSet.samples[0].texture!, options: [:])!
                sampleImage.saveJPEG("hi.jpg")
                let network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: layers, epochs: 1, batchSize: 40, numberOfClasses: 10)
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

