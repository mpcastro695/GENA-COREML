//
//  ContentView.swift
//  GENA Demo
//
//  Created by Martin Castro on 4/5/23.
//

import SwiftUI
import CoreML

struct ContentView: View {
    
    let model = try! GENA_FP16()
    @StateObject var tokenizer = BPTokenizer.loadTokenizer()!
    
    @State private var entry = String()
    @State private var results: [MLMultiArray] = []
    
    var body: some View {
        NavigationStack(path: $results) {
            VStack {
                HStack{
                    ZStack(alignment: .center){
                        Image("dsDNA")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 250)
                    }.frame(maxWidth: .infinity, maxHeight: .infinity)
                    Divider()
                    VStack(alignment: .leading){
                        Text("GENA +  Core ML")
                            .font(.largeTitle)
                            .bold()
                            .padding(.bottom, 5)
                        Text("GENA was trained with a masked-langauge modeling objective across the Telomere-to-Telomere (T2T) human genome assembly.")
                            .font(.callout)
                            .padding(.bottom)
                        Text("Enter a string of nucleotides to extract per-sequence and per k-mer features.")
                            .font(.callout)
                            .padding(.bottom)
                    }
                    .padding()
                    .frame(width: 300)
                    .foregroundColor(.secondary)
                            
                }
                .padding(10)
                
                TextEditor(text: $entry)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .frame(height: 150)
                
                Button {
                    let encodedInput = tokenizer.tokenize(entry)
                    guard let output = try? model.prediction(input_ids: encodedInput) else {
                            fatalError("Unexpected runtime error.")
                    }
                    print(output.features.shape)
                    results.append(output.features)
                } label: {
                    Text("Extract Features")
                }

            }
            .padding()
            .navigationDestination(for: MLMultiArray.self) { result in
                FeatureView(result: result, tokenizer: tokenizer)
            }
        }
        .frame(minWidth: 600, minHeight: 480)
        .environmentObject(tokenizer)
    }
    
    func convertToArray(from mlMultiArray: MLMultiArray) -> [Int32] {
        let length = mlMultiArray.count
        let intPointer = mlMultiArray.dataPointer.bindMemory(to: Int32.self, capacity: length)
        let intBuffer = UnsafeBufferPointer(start: intPointer, count: length)
        let output = Array(intBuffer)
        return output
    }
    
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
