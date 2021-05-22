//
//  DataSource.swift
//  GPUNN
//
//  Created by Евгений on 14.05.2021.
//

import Foundation
import MetalPerformanceShaders

class ConvDataSource: NSObject, MPSCNNConvolutionDataSource, Codable {
    let convolutionDescriptor: MPSCNNConvolutionDescriptor
    
    var weightsVector, weightsMomentumVector, weightsVelocityVector, biasVector, biasMomentumVector, biasVelocityVector: MPSVector
    
    let weightsPointer: UnsafeMutableRawPointer
    let biasPointer: UnsafeMutablePointer<Float>
    let _label: String
    let commandQueue: MTLCommandQueue
    
    var beta1, beta2: Double
    var epsilon: Float
    
    var learningRate: Float
    
    private enum CodingKeys: String, CodingKey {
        case beta1
        case beta2
        case epsilon
        case label
        case kernelSize
        case inputFC
        case outputFC
        case weightsVector
        case biasVector
        case weightsMomentum
        case biasMomentum
        case weightsVelocity
        case biasVelocity
        case learningRate
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(_label, forKey: .label)
        try container.encode(beta1, forKey: .beta1)
        try container.encode(beta2, forKey: .beta2)
        try container.encode(epsilon, forKey: .epsilon)
        try container.encode(CGSize(width: convolutionDescriptor.kernelWidth, height: convolutionDescriptor.kernelHeight), forKey: .kernelSize)
        try container.encode(convolutionDescriptor.inputFeatureChannels, forKey: .inputFC)
        try container.encode(convolutionDescriptor.outputFeatureChannels, forKey: .outputFC)
        try container.encode(weightsVector, forKey: .weightsVector)
        try container.encode(biasVector, forKey: .biasVector)
        try container.encode(weightsMomentumVector, forKey: .weightsMomentum)
        try container.encode(biasMomentumVector, forKey: .biasMomentum)
        try container.encode(weightsVelocityVector, forKey: .weightsVelocity)
        try container.encode(biasVelocityVector, forKey: .biasVelocity)
        try container.encode(learningRate, forKey: .learningRate)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        let device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        
        learningRate = try container.decode(Float.self, forKey: .learningRate)
        _label = try container.decode(String.self, forKey: .label)
        beta1 = try container.decode(Double.self, forKey: .beta1)
        beta2 = try container.decode(Double.self, forKey: .beta2)
        epsilon = try container.decode(Float.self, forKey: .epsilon)
        let kernelSize = try container.decode(CGSize.self, forKey: .kernelSize)
        let inputFC = try container.decode(Int.self, forKey: .inputFC)
        let outputFC = try container.decode(Int.self, forKey: .outputFC)
        convolutionDescriptor = .init(kernelWidth: Int(kernelSize.width), kernelHeight: Int(kernelSize.height), inputFeatureChannels: inputFC, outputFeatureChannels: outputFC)
        convolutionDescriptor.fusedNeuronDescriptor = .cnnNeuronDescriptor(with: .none)
        
        weightsVector = try container.decode(MPSVector.self, forKey: .weightsVector)
        biasVector = try container.decode(MPSVector.self, forKey: .biasVector)
        weightsMomentumVector = try container.decode(MPSVector.self, forKey: .weightsMomentum)
        biasMomentumVector = try container.decode(MPSVector.self, forKey: .biasMomentum)
        weightsVelocityVector = try container.decode(MPSVector.self, forKey: .weightsVelocity)
        biasVelocityVector = try container.decode(MPSVector.self, forKey: .biasVelocity)
        weightsPointer = weightsVector.data.contents()
        biasPointer = biasVector.data.contents().assumingMemoryBound(to: Float.self)
        
        convWeightsAndBiases = .init(weights: weightsVector.data, biases: biasVector.data)
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: learningRate, gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
        updater = MPSNNOptimizerAdam(device: device, beta1: beta1, beta2: beta2, epsilon: epsilon, timeStep: 0, optimizerDescriptor: optimizerDescriptor)
    }
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return convolutionDescriptor
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return weightsPointer
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return biasPointer
    }
    
    func checkpointWithCommandQueue(){
        autoreleasepool(invoking: {
            let commandBuffer = MPSCommandBuffer(from: commandQueue)
            convWeightsAndBiases.synchronize(on: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        })
    }
    
    func load() -> Bool {
        checkpointWithCommandQueue()
        return true
    }
    
    func purge() {
        return
    }
    
    func label() -> String? {
        return _label
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }
    
    func update(with commandBuffer: MTLCommandBuffer, gradientState: MPSCNNConvolutionGradientState, sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        updater.encode(commandBuffer: commandBuffer, convolutionGradientState: gradientState, convolutionSourceState: sourceState, inputMomentumVectors: [weightsMomentumVector, biasMomentumVector], inputVelocityVectors: [weightsVelocityVector, biasVelocityVector], resultState: convWeightsAndBiases)
        return convWeightsAndBiases
    }
    
    let updater: MPSNNOptimizerAdam
    let convWeightsAndBiases: MPSCNNConvolutionWeightsAndBiasesState
    
    init(device: MTLDevice, kernelWidth: Int, kernelHeight: Int, inputFeatureChannels: Int, outputFeatureChannnels: Int, stride: Int, learningRate: Float, commandQueue: MTLCommandQueue, num: Int) {
        _label = String(num)
        self.commandQueue = commandQueue
        self.learningRate = learningRate
        
        convolutionDescriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth, kernelHeight: kernelHeight, inputFeatureChannels: inputFeatureChannels, outputFeatureChannels: outputFeatureChannnels)
        convolutionDescriptor.strideInPixelsX = stride
        convolutionDescriptor.strideInPixelsY = stride
        convolutionDescriptor.fusedNeuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .none)
        
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-08;
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: learningRate, gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
        
        //updater = MPSNNOptimizerAdam(device: device, learningRate: learningRate)
        updater = MPSNNOptimizerAdam(device: device, beta1: beta1, beta2: beta2, epsilon: epsilon, timeStep: 0, optimizerDescriptor: optimizerDescriptor)
        
        let lenWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannnels
        let sizeWeights = lenWeights * MemoryLayout<Float32>.size
        let vDescWeights = MPSVectorDescriptor(length: lenWeights, dataType: .float32)
        
        let vDescBiases = MPSVectorDescriptor(length: outputFeatureChannnels, dataType: .float32)
        let sizeBiases = outputFeatureChannnels * MemoryLayout<Float32>.size
        
        var zero = Float.zero, biasInit = Float.zero
        
        weightsVector = MPSVector(device: device, descriptor: vDescWeights)
        weightsVelocityVector = MPSVector(device: device, descriptor: vDescWeights)
        weightsMomentumVector = MPSVector(device: device, descriptor: vDescWeights)
        weightsPointer = weightsVector.data.contents()
        let weightsVelocityPointer = weightsVelocityVector.data.contents()
        let weightsMomentumPointer = weightsMomentumVector.data.contents()
        
        memset_pattern4(weightsVelocityPointer, &zero, sizeWeights)
        memset_pattern4(weightsMomentumPointer, &zero, sizeWeights)
        
        biasVector = MPSVector(device: device, descriptor: vDescBiases)
        biasVelocityVector = MPSVector(device: device, descriptor: vDescBiases)
        biasMomentumVector = MPSVector(device: device, descriptor: vDescBiases)
        biasPointer = biasVector.data.contents().assumingMemoryBound(to: Float.self)
        let biasVelocityPointer = biasVelocityVector.data.contents()
        let biasMomentumPointer = biasMomentumVector.data.contents()
        
        memset_pattern4(biasPointer, &biasInit, sizeBiases)
        memset_pattern4(biasVelocityPointer, &zero, sizeBiases)
        memset_pattern4(biasMomentumPointer, &zero, sizeBiases)
        
        convWeightsAndBiases = .init(weights: weightsVector.data, biases: biasVector.data)
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
        
        let limit = sqrt(6.0 / Float(inputFeatureChannels + outputFeatureChannnels))
        let randomDescriptor = MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptor(withMinimum: -limit, maximum: limit)
        let randomKernel = MPSMatrixRandomMTGP32(device: device, destinationDataType: .float32, seed: 0, distributionDescriptor: randomDescriptor)
        randomKernel.encode(commandBuffer: commandBuffer, destinationVector: weightsVector)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        weightsVelocityVector.data.didModifyRange(0..<sizeWeights)
        weightsMomentumVector.data.didModifyRange(0..<sizeWeights)
        biasVector.data.didModifyRange(0..<sizeBiases)
        biasVelocityVector.data.didModifyRange(0..<sizeBiases)
        biasMomentumVector.data.didModifyRange(0..<sizeBiases)
    }
}

extension KeyedEncodingContainer {
    mutating func encode(_ value: MTLBuffer, forKey key: K) throws {
        let length = value.length
        let count = length / 4
        let result = value.contents().bindMemory(to: Float.self, capacity: count)
        var arr = Array(repeating: Float.zero, count: count)
        for i in 0..<count {
            arr[i] = result[i]
        }
        try encode(arr, forKey: key)
    }
    
    mutating func encode(_ value: MPSVector, forKey key: K) throws {
        try encode(value.data, forKey: key)
    }
}

extension KeyedDecodingContainer {
    func decode(_ type: MTLBuffer.Type, forKey key: K) throws -> MTLBuffer {
        var arr = try decode([Float].self, forKey: key)
        let length = arr.count * 4
        let device = MTLCreateSystemDefaultDevice()!
        let buffer = device.makeBuffer(bytes: arr, length: length, options: [])!
        return buffer
    }
    
    func decode(_ type: MPSVector.Type, forKey key: K) throws -> MPSVector {
        let buffer = try decode(MTLBuffer.self, forKey: key)
        let descriptor = MPSVectorDescriptor(length: buffer.length/4, dataType: .float32)
        return MPSVector(buffer: buffer, descriptor: descriptor)
    }
}
