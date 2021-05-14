//
//  DataSource.swift
//  GPUNN
//
//  Created by Евгений on 14.05.2021.
//

import Foundation
import MetalPerformanceShaders

class ConvDataSource {
    let convolution: MPSCNNConvolution
    init(device: MTLDevice, kernelWidth: Int, kernelHeight: Int, inputFeatureChannels: Int, outputFeatureChannnels: Int, stride: Int, learningRate: Float, commandQueue: MTLCommandQueue) {
        let convolutionDescriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth, kernelHeight: kernelHeight, inputFeatureChannels: inputFeatureChannels, outputFeatureChannels: outputFeatureChannnels)
        convolutionDescriptor.dilationRateX = stride
        convolutionDescriptor.dilationRateY = stride
        convolutionDescriptor.fusedNeuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .none)
        
        let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: learningRate, gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
        
        let updater = MPSNNOptimizerAdam(device: device, learningRate: learningRate)
        
        let randomDescriptor = MPSMatrixRandomDistributionDescriptor()
        randomDescriptor.maximum = 0.5
        randomDescriptor.minimum = -0.5
        
        let randomKernel = MPSMatrixRandomMTGP32(device: device, destinationDataType: .float32, seed: 0, distributionDescriptor: randomDescriptor)
        
        let lenWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannnels
        let vDescWeights = MPSVectorDescriptor(length: lenWeights, dataType: .float32)
        
        let vDescBiases = MPSVectorDescriptor(length: outputFeatureChannnels, dataType: .float32)
        
        let weightsVector = MPSVector(device: device, descriptor: vDescWeights)
        let weightsPointer = weightsVector.data.contents().assumingMemoryBound(to: Float.self)
        
        let biasVector = MPSVector(device: device, descriptor: vDescBiases)
        let biasPointer = biasVector.data.contents().assumingMemoryBound(to: Float.self)
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
        randomKernel.encode(commandBuffer: commandBuffer, destinationVector: weightsVector)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        convolution = MPSCNNConvolution(device: device, convolutionDescriptor: convolutionDescriptor, kernelWeights: weightsPointer, biasTerms: biasPointer, flags: .none)
    }
}
