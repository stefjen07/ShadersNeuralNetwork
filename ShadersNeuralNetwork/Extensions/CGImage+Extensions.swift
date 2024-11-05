//
//  CGImage+Extensions.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import CoreGraphics
import MetalPerformanceShaders
import Accelerate

extension CGImage {
    var texture: MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let texture = device.makeTexture(descriptor: descriptor)!
        let region = MTLRegion(origin: .init(x: 0, y: 0, z: 0), size: .init(width: width, height: height, depth: 1))
        
        var format = vImage_CGImageFormat(bitsPerComponent: UInt32(8), bitsPerPixel: UInt32(8), colorSpace: Unmanaged.passRetained(CGColorSpace(name: CGColorSpace.linearGray)!), bitmapInfo: .init(rawValue: CGImageAlphaInfo.none.rawValue), version: 0, decode: nil, renderingIntent: .defaultIntent)
        do {
            var sourceBuffer = try vImage_Buffer(cgImage: self, format: format)
            var error = vImage_Error()
            let destImage = vImageCreateCGImageFromBuffer(&sourceBuffer, &format, nil, nil, numericCast(kvImageNoFlags), &error).takeRetainedValue()
            
            guard error == noErr else {
                fatalError()
            }
            
            let dstData = destImage.dataProvider?.data
            let pixelData = CFDataGetBytePtr(dstData!)
            
            texture.replace(region: region, mipmapLevel: 0, withBytes: pixelData!, bytesPerRow: bytesPerRow)
            
            return texture
        } catch {
            fatalError(error.localizedDescription)
        }
    }
    
    func grayscale(size: CGSize) -> CGImage {
        let imageRect:CGRect = CGRect(origin: .zero, size: size)
        let colorSpace = CGColorSpace(name: CGColorSpace.linearGray)!
        let width = size.width
        let height = size.height
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let context = CGContext(data: nil, width: Int(width), height: Int(height), bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
        context?.draw(self, in: imageRect)
        if let makeImg = context?.makeImage() {
            return makeImg
        }
        return self
    }
    
    var png: NSData? {
        guard let mutableData = CFDataCreateMutable(nil, 0),
              let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { print("Unable to get PNG data"); return nil }
        CGImageDestinationAddImage(destination, self, nil)
        guard CGImageDestinationFinalize(destination) else { print("Unable to get PNG data"); return nil }
        return mutableData as NSData
    }
}
