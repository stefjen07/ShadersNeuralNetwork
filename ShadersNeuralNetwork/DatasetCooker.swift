//
//  DatasetCooker.swift
//  GPUNN
//
//  Created by Евгений on 14.05.2021.
//

import Foundation
import MetalPerformanceShaders

func cook(device: MTLDevice) {
    let currentUrl = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let dataset = Dataset(device: device, folderUrl: currentUrl.appendingPathComponent("Set"), imageSize: CGSize(width: 32, height: 32))
    do {
        try dataset.save(to: currentUrl.appendingPathComponent("set.ds"))
    } catch {
        
    }
}

func getDS() -> Dataset {
    let currentUrl = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    do {
        return try Dataset(from: currentUrl.appendingPathComponent("set.ds"))
    } catch {
        
    }
    return Dataset()
}
