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

enum Classifier: String {
    case nlp
    case hiragana
    case mnist
    
    var performer: BasePerformer {
        switch self {
        case .nlp:
            NLPPerformer(learningRate: learningRate)
        case .hiragana:
            HiraganaPerformer(learningRate: learningRate)
        case .mnist:
            MNISTPerformer(learningRate: learningRate)
        }
    }
}

if let rawValue = ProcessInfo.processInfo.environment["classifier"] {
    Classifier(rawValue: rawValue)?.performer.perform()
}
