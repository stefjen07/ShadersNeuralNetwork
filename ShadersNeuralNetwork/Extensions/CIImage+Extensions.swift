//
//  CIImage+Extensions.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import CoreImage

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
    
    var convertedCGImage: CGImage? {
        let context = CIContext(options: nil)
        return context.createCGImage(self, from: self.extent)
    }
    
    var inverted: CIImage {
        let filter = CIFilter(name: "CIColorInvert")!
        filter.setValue(self, forKey: kCIInputImageKey)
        
        return filter.outputImage ?? self
    }
    
    func resize(targetSize: CGSize) -> CIImage {
        let resizeFilter = CIFilter(name:"CILanczosScaleTransform")!

        let scale = targetSize.height / self.extent.height
        let aspectRatio = targetSize.width/(self.extent.width * scale)

        resizeFilter.setValue(self, forKey: kCIInputImageKey)
        resizeFilter.setValue(scale, forKey: kCIInputScaleKey)
        resizeFilter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        return resizeFilter.outputImage ?? self
    }
}
