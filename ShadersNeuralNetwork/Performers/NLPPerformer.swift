//
//  NLPPerformer.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import Foundation

class NLPPerformer: BasePerformer {
    let inputSize = 15
    
    override var trainDatasetURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("textTrain.ds")
    }
    
    override var testDatasetURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("textTest.ds")
    }
    
    override var modelURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("nlp.nnm")
    }
    
    private var shakespeareURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("shakespeare.html")
    }
    
    private var tokensURL: URL {
        URL.currentDirectoryURL.appendingPathComponent("tokenizer.st")
    }
    
    var isTokenized: Bool {
        FileManager.default.fileExists(atPath: tokensURL.absoluteString)
    }
    
    override func perform() {
        if !isTokenized {
            let text = try? String(contentsOf: shakespeareURL)
            guard let text = text else {
                fatalError()
            }
            let tokenizer = Tokenizer(text: text)
            var set = tokenizer.getDataset(inputSize: inputSize)
            set.optimize()
            var trainSet = Dataset(), testSet = Dataset()
            set.breakInto(trainSet: &trainSet, evaluationSet: &testSet, evaluationPart: 0.2)
            try? trainSet.save(to: trainDatasetURL)
            try? testSet.save(to: testDatasetURL)
            try? tokenizer.save(to: tokensURL)
        }
        let trainSet = try? Dataset(from: trainDatasetURL)
        let testSet = try? Dataset(from: testDatasetURL)
        let tokenizer = try? Tokenizer(from: tokensURL)
        guard let trainSet, let testSet, let tokenizer else {
            fatalError()
        }
        
        let layers: [Layer] = [
            Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 2*inputSize, outputFC: 128, stride: 1, learningRate: learningRate),
            ReLU(),
            Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC:128, outputFC: 512, stride: 1, learningRate: learningRate),
            ReLU(),
            Dense(device: device, commandQueue: commandQueue, kernelSize: .init(width: 1, height: 1), inputFC: 512, outputFC: 256, stride: 1, learningRate: learningRate)
        ]
        var network = NeuralNetwork(device: device, commandQueue: commandQueue, layers: layers, epochs: 5, batchSize: 32, numberOfClasses: 256)
        if modelExists {
            do {
                network = try NeuralNetwork(from: modelURL)
            } catch {
                
            }
        }
        network.train(trainSet: trainSet, evaluationSet: testSet)
        try? network.save(to: modelURL)
        var input = "Romeo and Juliet is the Shakespeare's book. You should read it before writing neural networks. Romeo"
        let additionSize = 20
        for _ in 0..<additionSize {
            var parts = input.split(separator: " ")
            var inputArr: [String] = []
            for _ in 0..<inputSize {
                inputArr.append(String(parts.last!))
                parts.removeLast()
            }
            let sample = tokenizer.encodeSample(words: inputArr)
            let newWord = tokenizer.getWord(id: network.predict(samples: [sample]).first!)
            input =  "\(input) \(newWord)"
        }
        print(input)
    }
}
