//
//  URL+Extensions.swift
//  GPUNN
//
//  Created by Yauheni Stsefankou on 05.11.2024.
//

import Foundation

extension URL {
    static var currentDirectoryURL: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }
}
