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
    var dataset = Dataset(
        device: device, 
        folderUrl: URL.currentDirectoryURL.appendingPathComponent("Set"),
        imageSize: CGSize(width: 32, height: 32)
    )
    dataset.updateImageSize()
    
    do {
        try dataset.save(to: URL.currentDirectoryURL.appendingPathComponent("set.ds"))
    } catch {
        print(error.localizedDescription)
    }
}

func getDS() -> Dataset {
    do {
        return try Dataset(from: URL.currentDirectoryURL.appendingPathComponent("set.ds"))
    } catch {
        print(error.localizedDescription)
    }
    return Dataset()
}

class MNISTDataset: NSObject {
    var set: Dataset = .init()
    private static let baseURL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    private static let names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte" , "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    let imagePrefixLength = 16
    let labelPrefixLength = 8
    var isTrain = false
    init(isTrain: Bool = true) {
        self.isTrain = isTrain
    }
    
    private func downloadFile(name: String, completion: @escaping (Data?)->Void) throws {
        let path = URL.currentDirectoryURL.appendingPathComponent(name).path
        if FileManager.default.fileExists(atPath: path) {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
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
                    print(error?.localizedDescription ?? "")
                    completion(nil)
                }
            }.resume()
        }
    }
    
    var imageData: Data!
    var labelData: Data!
    
    func load() throws {
        let imageFileName = Self.names[isTrain ? 0 : 2]
        let labelFileName = Self.names[isTrain ? 1 : 3]
        let group = DispatchGroup()
        group.enter()
        try downloadFile(name: imageFileName) { [weak self] (data) in
            self?.imageData = data
            group.leave()
        }
        group.enter()
        try downloadFile(name: labelFileName) { [weak self] data in
            self?.labelData = data
            group.leave()
        }
        group.wait()
        
        if imageData == nil || labelData == nil {
            fatalError("Unable to download MNIST dataset")
        }
    }
    
    var count: Int {
        return (imageData.count - imagePrefixLength) / (28 * 28)
    }
    
    func fillSet() {
        var start = 0
        var end = 0
        
        let dummySample = DataSample(bytes: [], label: 0)
        set.samples = Array(repeating: dummySample, count: count)
        
        DispatchQueue.concurrentPerform(iterations: count, execute: { index in
            start = imagePrefixLength + index * 28 * 28
            end = start + 28*28
            let label = Int(labelData[labelPrefixLength + index])
            
            var data = Data(self.imageData[start..<end].map { 255 - $0 })
            data.withUnsafeMutableBytes {
                let context = CGContext(
                    data: $0.baseAddress,
                    width: 28,
                    height: 28,
                    bitsPerComponent: 8,
                    bytesPerRow: 28,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: 0
                )
                
                if let image = context?.makeImage() {
                    set.samples[index] = DataSample(image: image, label: label)
                }
            }
        })
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
