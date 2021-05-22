//
//  TextPreparing.swift
//  GPUNN
//
//  Created by Евгений on 22.05.2021.
//

import Foundation

struct Tokenizer: Codable {
    var tokens: [String]
    var ids: [Int]
    
    func encodeSample(words: [String]) -> DataSample {
        var ids: [UInt8] = []
        for word in words {
            var num = getId(word: word)
            var array = [UInt8]()
            while num > 0 {
                array.append(UInt8(num & 0xff))
                num >>= 8
            }
            if array.count < 1 {
                array = [0]
            }
            if array.count < 2 {
                array = [0, array[0]]
            }
            ids.append(contentsOf: array)
        }
        return DataSample(bytes: ids, label: -1)
    }
    
    init(from url: URL) throws {
        let decoder = JSONDecoder()
        let data = try Data(contentsOf: url)
        let tokenizer = try decoder.decode(Tokenizer.self, from: data)
        tokens = tokenizer.tokens
        ids = tokenizer.ids
    }
    
    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
    
    func translate() -> String {
        fatalError()
    }
    
    func getDataset(inputSize: Int) -> Dataset {
        var bytes = [[UInt8]]()
        var labels = [Int]()
        
        var i = inputSize
        
        while i < ids.count {
            var input = [UInt8]()
            for j in i-inputSize..<i {
                var num = ids[j]
                var array = [UInt8]()
                while num > 0 {
                    array.append(UInt8(num & 0xff))
                    num >>= 8
                }
                if array.count < 1 {
                    array = [0]
                }
                if array.count < 2 {
                    array = [0, array[0]]
                }
                input.append(contentsOf: array)
            }
            bytes.append(input)
            labels.append(ids[i])
            i += 1
        }
        
        return Dataset(bytes: bytes, labels: labels, classLabels: tokens)
    }
    
    func getWord(id: Int) -> String {
        if tokens.count > id {
            return tokens[id]
        }
        return ""
    }
    
    func getId(word: String) -> Int {
        for i in 0..<tokens.count {
            if tokens[i] == word {
                return i
            }
        }
        return 0
    }
    
    init(text: String) {
        let textParts = text.split(whereSeparator: { c in
            return c == " " || c == "\n"
        })
        var words = [String]()
        for part in textParts {
            words.append(String(part))
        }
        tokens = []
        ids = []
        for word in words {
            var found = false
            for i in 0..<tokens.count {
                if tokens[i] == word {
                    ids.append(i)
                    found = true
                    break
                }
            }
            if !found {
                tokens.append(word)
            }
        }
    }
}
