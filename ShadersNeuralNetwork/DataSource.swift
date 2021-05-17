//
//  DataSource.swift
//  GPUNN
//
//  Created by Евгений on 14.05.2021.
//

import Foundation
import MetalPerformanceShaders

class ConvDataSource: NSObject, MPSCNNConvolutionDataSource {
    let convolutionDescriptor: MPSCNNConvolutionDescriptor
    
    var weightsVector, weightsMomentumVector, weightsVelocityVector, biasVector, biasMomentumVector, biasVelocityVector: MPSVector
    
    let weightsPointer: UnsafeMutableRawPointer
    let biasPointer: UnsafeMutablePointer<Float>
    let _label: String
    let commandQueue: MTLCommandQueue
    
    var beta1, beta2: Double
    var epsilon: Float
    var t: Int
    
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
        t+=1
        updater.encode(commandBuffer: commandBuffer, convolutionGradientState: gradientState, convolutionSourceState: sourceState, inputMomentumVectors: [weightsMomentumVector, biasMomentumVector], inputVelocityVectors: [weightsVelocityVector, biasVelocityVector], resultState: convWeightsAndBiases)
        //assert(t == updater.timeStep)
        return convWeightsAndBiases
    }
    
    let updater: MPSNNOptimizerAdam
    let convWeightsAndBiases: MPSCNNConvolutionWeightsAndBiasesState
    
    init(device: MTLDevice, kernelWidth: Int, kernelHeight: Int, inputFeatureChannels: Int, outputFeatureChannnels: Int, stride: Int, learningRate: Float, commandQueue: MTLCommandQueue, num: Int) {
        _label = String(num)
        self.commandQueue = commandQueue
        
        convolutionDescriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth, kernelHeight: kernelHeight, inputFeatureChannels: inputFeatureChannels, outputFeatureChannels: outputFeatureChannnels)
        convolutionDescriptor.strideInPixelsX = stride
        convolutionDescriptor.strideInPixelsY = stride
        convolutionDescriptor.fusedNeuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .none)
        
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-08;
        t = 0
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: learningRate, gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
        
        //updater = MPSNNOptimizerAdam(device: device, learningRate: learningRate)
        updater = MPSNNOptimizerAdam(device: device, beta1: beta1, beta2: beta2, epsilon: epsilon, timeStep: 0, optimizerDescriptor: optimizerDescriptor)
        
        let randomDescriptor = MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptor(withMinimum: -0.5, maximum: 0.5)
        let randomKernel = MPSMatrixRandomMTGP32(device: device, destinationDataType: .float32, seed: 0, distributionDescriptor: randomDescriptor)
        
        let lenWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannnels
        let sizeWeights = lenWeights * MemoryLayout<Float32>.size
        let vDescWeights = MPSVectorDescriptor(length: lenWeights, dataType: .float32)
        
        let vDescBiases = MPSVectorDescriptor(length: outputFeatureChannnels, dataType: .float32)
        let sizeBiases = outputFeatureChannnels * MemoryLayout<Float32>.size
        
        var zero = Float.zero, biasInit = Float(0.1)
        //let &zero: UnsafeMutablePointer = UnsafeMutableRawPointer(&zero).assumingMemoryBound(to: UInt8.self),
            //biasInitPointer = UnsafeMutableRawPointer(&biasInit).assumingMemoryBound(to: UInt8.self)
        
        weightsVector = MPSVector(device: device, descriptor: vDescWeights)
        weightsVelocityVector = MPSVector(device: device, descriptor: vDescWeights)
        weightsMomentumVector = MPSVector(device: device, descriptor: vDescWeights)
        weightsPointer = weightsVector.data.contents()
        let weightsVelocityPointer = weightsVelocityVector.data.contents().assumingMemoryBound(to: Float.self)
        let weightsMomentumPointer = weightsMomentumVector.data.contents().assumingMemoryBound(to: Float.self)
        
        memset_pattern4(weightsVelocityPointer, &zero, sizeWeights)
        memset_pattern4(weightsMomentumPointer, &zero, sizeWeights)
        
        biasVector = MPSVector(device: device, descriptor: vDescBiases)
        biasVelocityVector = MPSVector(device: device, descriptor: vDescBiases)
        biasMomentumVector = MPSVector(device: device, descriptor: vDescBiases)
        biasPointer = biasVector.data.contents().assumingMemoryBound(to: Float.self)
        let biasVelocityPointer = biasVelocityVector.data.contents().assumingMemoryBound(to: Float.self)
        let biasMomentumPointer = biasMomentumVector.data.contents().assumingMemoryBound(to: Float.self)
        
        
        
        memset_pattern4(biasPointer, &biasInit, sizeBiases)
        memset_pattern4(biasVelocityPointer, &zero, sizeBiases)
        memset_pattern4(biasMomentumPointer, &zero, sizeBiases)
        
        convWeightsAndBiases = .init(weights: weightsVector.data, biases: biasVector.data)
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
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
