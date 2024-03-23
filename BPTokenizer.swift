//
//  BPTokenizer.swift
//  GENA Demo
//
//  Created by Martin Castro on 12/30/23.
//

import Foundation
import CoreML

class BPTokenizer: Decodable, ObservableObject {
    
    static let maxTokens = 512
    static let overheadTokens = 2
    let vocab: [String:Int]
    let merges: [String]
    
    @Published var tokenCount = 0
    @Published var tokenSequence: [String] = []
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let nestedContainer = try container.nestedContainer(keyedBy: CodingKeys.self, forKey: .model)
        vocab = try nestedContainer.decode([String:Int].self, forKey: .vocab)
        merges = try nestedContainer.decode([String].self, forKey: .merges)
    }
    enum CodingKeys: String, CodingKey {
        case model = "model"
        case vocab = "vocab"
        case merges = "merges"
    }
    
    static func loadTokenizer() -> BPTokenizer? {
        if let url = Bundle.main.url(forResource: "vocab", withExtension: "json") {
            do {
                let data = try Data(contentsOf: url)
                let decoder = JSONDecoder()
                let tokenizer = try decoder.decode(BPTokenizer.self, from: data)
                return tokenizer
            } catch {
                print("error:\(error)")
            }
        }
        return nil
    }
    
    private func mergeKmerPairs(in sequence: [String], _ a: String, _ b: String, with c: String) -> [String] {
        var result = [String]()
        var index = 0
        while index < sequence.count {
            if (index + 1 < sequence.count) && (sequence[index] == a) && (sequence[index + 1] == b) {
                result.append(c)
                index += 2 // Skip the pair
            } else {
                result.append(sequence[index])
                index += 1
            }
        }
        return result
    }

    public func tokenize(_ sequence: String) -> MLShapedArray<Int32> {
        let cleanedSeq = sequence.filter { !$0.isWhitespace }
        var tokens = Array(cleanedSeq).compactMap{String($0)}
        for merge in merges {
            let mergingPair = merge.split(separator: " ")
            tokens = mergeKmerPairs(in: tokens, String(mergingPair[0]), String(mergingPair[1]), with: mergingPair.joined())
        }
        tokens = Array(tokens.prefix(BPTokenizer.maxTokens-BPTokenizer.overheadTokens))
        tokens.insert("[CLS]", at: 0)
        
        tokenSequence = tokens
        tokenCount = tokens.count
        
        tokens.append("[SEP]")
        // Fill remaining slots with padding tokens
        let padding = BPTokenizer.maxTokens - tokens.count
        tokens += Array(repeating: "[PAD]", count: padding)
        
        let vocabIDs = tokens.compactMap{vocab[$0]}
        
        return MLShapedArray(scalars: vocabIDs.compactMap{Int32($0)}, shape: [1, BPTokenizer.maxTokens])
    }
    
}
