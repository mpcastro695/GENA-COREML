//
//  FeatureMap.swift
//  ProtBERT Demo
//
//  Created by Martin Castro on 4/8/23.
//

import SwiftUI
import CoreML

struct FeatureView: View {
    
    var result: MLMultiArray
    @ObservedObject var tokenizer: BPTokenizer
    
    private let featureSize = 768
    private let singleRow = [GridItem()]
    
    var body: some View{
        VStack(alignment: .leading) {
            VStack(alignment: .leading){
                Text("Features")
                    .font(.title)
                    .bold()
                Text("Sequence length: \(tokenizer.tokenCount - 1)")
            }
            .foregroundColor(.secondary)
            .padding(.top)
            .padding(.horizontal)
            ScrollView(.horizontal){
                LazyHGrid(rows: singleRow) {
                    ForEach(Array(getTokenSlices(for: result, tokenCount: tokenizer.tokenCount).enumerated()), id: \.offset) { index, tokenSlice in
                        VStack{
                            Features2DVisual(values: Array(tokenSlice))
                            Text("\(tokenizer.tokenSequence[index])")
                                .font(.headline)
                                .bold()
                            Text("Index: \(index)")
                        }
                        .padding(.horizontal, 15)
                    }
                }
            }.frame(minHeight: 240)
        }
    }
    
    private func convertToArray(from mlMultiArray: MLMultiArray) -> [Float32] {
        let length = mlMultiArray.count
        let floatPointer = mlMultiArray.dataPointer.bindMemory(to: Float32.self, capacity: length)
        let floatBuffer = UnsafeBufferPointer(start: floatPointer, count: length)
        let output = Array(floatBuffer)
        return output
    }
    
    private func getTokenSlices(for multiArray: MLMultiArray, tokenCount: Int) -> [ArraySlice<Float32>] {
        var tokenSlices: [ArraySlice<Float32>] = []
        let flattenedArray = convertToArray(from: multiArray)
        
        var startIndice = 0
        var endIndice = featureSize - 1
        for _ in 0..<tokenCount {
            // Grab the appropriate slice of values (e.g., 0..767, 768..1535, ...)
            let slice = flattenedArray[startIndice...endIndice]
            tokenSlices.append(slice)
            startIndice += featureSize
            endIndice += featureSize
        }
        return tokenSlices
    }
}


struct Features2DVisual: View {
    
    let gridSize = CGSize(width: 6, height: 6)
    let spacing: CGFloat = 2
    let lineWidth: CGFloat = 0.2
    let height = 32
    let width = 24 // height * width = 768 features per token
    let posLimit: Double = 6
    let negLimit: Double = -6
    
    let values: [Float32]
    
    var body: some View {
        Canvas { context, size in
            var x = 0
            var y = 0
            for i in 0..<(height*width){
                
                let origin = CGPoint(x: CGFloat(x)*(gridSize.width + spacing), y: CGFloat(y)*(gridSize.height + spacing))
                let rect = CGRect(origin: origin, size: gridSize)
                let rectPath = Path(rect)
                
                //Calculate cell color
                let value = values[i]
                if value > Float32(posLimit){ // Shade Green
                    let color = Color.green
                    context.fill(rectPath, with: .color(color))
                }else if value >= 0 { // Shade Green
                    let color = Color.green.opacity(Double(value)/posLimit)
                    context.fill(rectPath, with: .color(color))
                }else if value < Float32(negLimit){ // Shade red
                    let color = Color.red
                    context.fill(rectPath, with: .color(color))
                }else{ // Shade red
                    let color = Color.red.opacity(Double(value)/negLimit)
                    context.fill(rectPath, with: .color(color))
                }
                context.stroke(rectPath, with: .color(.secondary), lineWidth: lineWidth)
                
                // Increment position, L -> R, T -> B
                x += 1
                if x >= width {
                    x -= width
                    y += 1
                }
            }
        }
        .frame(width:200, height: 240)
    }
}

struct Feature2DVisial_Previews: PreviewProvider {

    static var previews: some View {
        let randomNumbers: [Float32] = (1...784).map { _ in Float32.random(in: -8...4) }
        Features2DVisual(values: randomNumbers)
    }
}
