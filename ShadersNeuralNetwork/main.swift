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

enum Classifier {
    case npl
    case hiragana
    case mnist
}

let classifier: Classifier = .npl

switch classifier {
case .hiragana:
    performHiragana(learningRate: learningRate, firstTime: false, fromFile: true)
case .mnist:
    performMNIST(learningRate: learningRate, firstTime: false, fromFile: false)
case .npl:
    performNPL(learningRate: learningRate, firstTime: true, fromFile: false)
}
