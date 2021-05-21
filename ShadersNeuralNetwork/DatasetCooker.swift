//
//  DatasetCooker.swift
//  GPUNN
//
//  Created by Евгений on 14.05.2021.
//

import Foundation
import MetalPerformanceShaders
import CoreImage

func cook(device: MTLDevice) {
    let currentUrl = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var dataset = Dataset(device: device, folderUrl: currentUrl.appendingPathComponent("Set"), imageSize: CGSize(width: 32, height: 32))
    dataset.updateImageSize()
    do {
        try dataset.save(to: currentUrl.appendingPathComponent("set.ds"))
    } catch {
        fatalError("Error")
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

class MNISTDataset {
    var set: Dataset = .init()
    private static let baseURL = "http://yann.lecun.com/exdb/mnist/"
    private static let names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte" , "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    let imagePrefixLength = 16
    let labelPrefixLength = 8
    var isTrain = false
    init(isTrain: Bool = true) {
        self.isTrain = isTrain
    }
    
    func dataPath(with name: String) -> String {
        return FileManager.default.currentDirectoryPath + "/" + name
    }
    
    private func downloadFile(name: String, completion: @escaping (Data)->Void) throws {
        let path = dataPath(with: name)
        if FileManager.default.fileExists(atPath: path) {
            let data =  try Data(contentsOf: URL(fileURLWithPath: path))
            completion(data)
        } else {
            let url = URL(string: "\(Self.baseURL)\(name).gz")!
            print("Downloading \(url.absoluteString)")
            let request = URLRequest(url: url, timeoutInterval: 3600)
            URLSession.shared.dataTask(with: request) { (data, response, error) in
                if error == nil, let data = data {
                    FileManager.default.createFile(atPath: path + ".gz", contents: data, attributes: nil)
                    _ = execute(launchPath: gzipPath(), currentDirectory: Bundle.main.bundlePath, arguments: ["-d", path + ".gz"])
                    print("Save to \(path)")
                    let data =  try! Data(contentsOf: URL(fileURLWithPath: path))
                    completion(data)
                } else {
                    print("Error:\(error)")
                }
            }.resume()
            
        }
    }
    
    var imageData: Data!
    var labelData: Data!
    
    func load() throws {
        let imageIndex = isTrain ? 0 : 2
        let labelIndex = isTrain ? 1 : 3
        let imageFileName = Self.names[imageIndex]
        let labelFileName = Self.names[labelIndex]
        let group = DispatchGroup()
        group.enter()
        try downloadFile(name: imageFileName) { [weak self] (data) in
            self?.imageData = data
            group.leave()
        }
        group.wait()
        group.enter()
        try downloadFile(name: labelFileName, completion: { [weak self] data in
            self?.labelData = data
            group.leave()
        })
        group.wait()
    }
    
    var count: Int {
        return (imageData.count - imagePrefixLength) / (28 * 28)
    }
    
    func fillSet() {
        var start = 0
        var end = 0
        for index in 0..<count {
            start = imagePrefixLength + index * 28 * 28
            end = start + 28*28
            let label = Int(labelData[labelPrefixLength + index])
            
            if let cgImage = CIImage(bitmapData: imageData[start..<end], bytesPerRow: 28, size: .init(width: 28, height: 28), format: .R8, colorSpace: .init(name: CGColorSpace.linearGray)!).inverted.convertedCGImage {
                let sample = DataSample(image: cgImage, label: label)
                set.samples.append(sample)
            }
        }
    }
}

func execute(launchPath: String, currentDirectory: String? = nil, arguments: [String] = []) -> String {
    let pipe = Pipe()
    let file = pipe.fileHandleForReading
    let task = Process()
    task.launchPath = launchPath
    task.arguments = arguments
    task.standardOutput = pipe
    if let currentDirectory = currentDirectory  {
        task.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
    }
    task.launch()
    let data = file.readDataToEndOfFile()
    return String(data: data, encoding: .utf8)!
}

func gzipPath() -> String {
    return execute(launchPath: "/usr/bin/which", arguments: ["gzip"]).components(separatedBy: .newlines).first!
}
