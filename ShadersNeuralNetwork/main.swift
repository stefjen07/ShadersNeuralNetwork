//
//  main.swift
//  GPUNN
//
//  Created by Евгений on 11.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

let learningRate: Float = 0.001

performHiragana(learningRate: learningRate)
//performMNIST(learningRate: learningRate)

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
