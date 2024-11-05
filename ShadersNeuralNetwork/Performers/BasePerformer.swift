//
//  NNPerformer.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import MetalPerformanceShaders

class BasePerformer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var learningRate: Float
    
    var trainDatasetURL: URL {
        fatalError("Must be overriden")
    }
    
    var testDatasetURL: URL {
        fatalError("Must be overriden")
    }
    
    var modelURL: URL {
        fatalError("Must be overriden")
    }
    
    var isDatasetPrepared: Bool {
        FileManager.default.fileExists(atPath: trainDatasetURL.path) &&
        FileManager.default.fileExists(atPath: testDatasetURL.path)
    }
    
    var modelExists: Bool {
        FileManager.default.fileExists(atPath: modelURL.path)
    }
    
    init(learningRate: Float) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Unable to get GPU device.")
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Unable to get command queue.")
        }
        
        self.device = device
        self.commandQueue = commandQueue
        self.learningRate = learningRate
    }
    
    func perform() {}
}
