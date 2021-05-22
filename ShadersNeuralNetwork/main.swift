//
//  main.swift
//  GPUNN
//
//  Created by Евгений on 11.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

fileprivate let learningRate: Float = 1e-4

//performNPL(learningRate: learningRate)
performHiragana(learningRate: learningRate, firstTime: false, fromFile: true)
//performMNIST(learningRate: learningRate)
