//
//  ContentView.swift
//  YoloV8
//
//

import SwiftUI
import UIKit
import Vision
struct ContentView: View {
    
    @State var testImageSrc : UIImage? = UIImage(contentsOfFile: Bundle.main.path(forResource: "tomcruise", ofType: "jpeg")!) ?? nil
    
    @State var maxProbValue:String = ""
    @State var bestMaskIdx = 0
    let imageViewWidth = CGFloat(678/2)
    let imageViewHeight = CGFloat(452/2)
    var body: some View {
        VStack {
            Image(uiImage: testImageSrc!)
                .resizable()
                .frame(width: imageViewWidth,height: imageViewHeight)
                .scaledToFill()
                .padding(EdgeInsets.init(top: 50, leading: 10, bottom: 30, trailing: 10))
            Text("Tom Cruise Prob: \(maxProbValue)")
                .padding()
            Button("Run Inference"){
                //ClassifiyImageUsingVN()
                ClassifyImage()
               
            }.padding(EdgeInsets.init(top: 20, leading: 10, bottom: 30, trailing: 10))
        }
        
    }
    
    func ClassifyImage(){
        let results =  YoloClassifier().classifyImage()!
        print(results)
        let boundingBox = getBoundingBox(feature:results.var_1504)
        DrawMask(boundingBox, masks: results.p)
    }
    
    
    fileprivate func DrawMask(_ boundingBox: CGRect, masks: MLMultiArray) {
        let testImage = UIImage(contentsOfFile: Bundle.main.path(forResource: "tomcruise", ofType: "jpeg")!)!
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: imageViewWidth, height: imageViewHeight))
        
        let scaledX : CGFloat = (boundingBox.minX/650)*imageViewWidth
        let scaledY : CGFloat = (boundingBox.minY/650)*imageViewHeight
        let scaledWidth : CGFloat = (boundingBox.width/650)*imageViewWidth
        let scaledHeight : CGFloat = (boundingBox.height/650)*imageViewHeight
        
        let rectangle = CGRect(x: scaledX, y: scaledY, width: scaledWidth, height: scaledHeight)
        print("scaled rectangle \(rectangle)")
        
        
        let maskProbThreshold : Float = 0.5
        let maskFill : Float = 1.0
        //draw the mask
        var maskProbalities : [[Float]] = [] //this will contains 160x160 mask pixel probablities
        var maskProbYAxis : [Float] = []
        print("Actual Image bounds \(rectangle)")
        //get the bounds for mask to match the bounds
        let mask_x_min = (rectangle.minX/imageViewWidth)*160
        let mask_x_max = (rectangle.maxX/imageViewWidth)*160
        
        let mask_y_min = (rectangle.minY/imageViewHeight)*160
        let mask_y_max = (rectangle.maxY/imageViewHeight)*160
        
        for y in 0..<masks.shape[2].intValue{
            maskProbYAxis.removeAll()
            for x in 0..<masks.shape[3].intValue{
                let pointKey = [0,bestMaskIdx,y,x] as [NSNumber]
                if(sigmoid(z: masks[pointKey].floatValue) < maskProbThreshold
                   && x >=  Int(mask_x_min) && x <= Int(mask_x_max)
                && y >= Int(mask_y_min) && y <= Int(mask_y_max)){
                    maskProbYAxis.append(1.0)
                }
                else{
                    maskProbYAxis.append(0.0)
                }
            }
            maskProbalities.append(maskProbYAxis)
        }
        
        let mask = renderer.image(){ context in
            
            context.cgContext.setLineWidth(1)
            for y in 0..<maskProbalities.count {
                for x in 0..<maskProbalities[y].count{
                    
                    let xFactor = Float(imageViewWidth)/160
                    let yFactor = Float(imageViewHeight)/160
                    let maskScaled_X = Double(x) * Double(xFactor)
                    let maskScaled_Y = Double(y) * Double(yFactor)
                    
                    if(maskProbalities[y][x] == 1.0)
                    {
                        context.cgContext.setStrokeColor(UIColor.red.withAlphaComponent(0.2).cgColor)
                        context.cgContext.addRect(CGRect(x: maskScaled_X, y:maskScaled_Y , width: 1, height: 1))
                        context.cgContext.drawPath(using: .stroke)
                    }
                }
            }
        }
        
        let imageWithBox = renderer.image(){ context in
            testImage.draw(in: CGRect(x: 0, y: 0, width: imageViewWidth, height: imageViewHeight))
            //context.cgContext.draw(testImage.cgImage!, in: )
            context.cgContext.setShouldAntialias(true)
            context.cgContext.setStrokeColor(UIColor.red.cgColor)
            context.cgContext.setLineWidth(2)
            context.cgContext.addRect(rectangle)
            context.cgContext.drawPath(using: .stroke)
            mask.draw(in: CGRect(x: 0, y: 0, width: imageViewWidth, height: imageViewHeight))
        }
         
        self.testImageSrc = imageWithBox
    }
    
    private func sigmoid(z:Float) -> Float{
        return 1.0/(1.0+exp(z))
    }
    
    func ClassifiyImageUsingVN(){
        YoloClassifier().classifyVisionModel(onClassified: { results in
            print(results)
            let result_0 = results[0] as! VNCoreMLFeatureValueObservation
            let result_1 = results[1] as! VNCoreMLFeatureValueObservation
            print(result_0.featureName)
            print(result_1.featureName)
            let boundingBox = getBoundingBox(feature:result_1.featureValue.multiArrayValue!)
            
            DrawMask(boundingBox,masks: result_1.featureValue.multiArrayValue! )
        })
    }
    
    func getBoundingBox(feature: MLMultiArray)->CGRect{
        var boundingBox = CGRect(x: 0,y: 0,width: 10,height: 10)
        
        var probMaxIdx = 0
        var maxProb : Float = 0
        var box_x : Float = 0
        var box_y : Float = 0
        var box_width : Float = 0
        var box_height : Float = 0
        
        for j in 0..<feature.shape[2].intValue-1
        {
            let key = [0,4,j] as [NSNumber]
            let nextKey = [0,4,j+1] as [NSNumber]
            if(feature[key].floatValue < feature[nextKey].floatValue){
                if(maxProb < feature[nextKey].floatValue){
                    probMaxIdx = j+1
                    let xKey = [0,0,probMaxIdx] as [NSNumber]
                    let yKey = [0,1,probMaxIdx] as [NSNumber]
                    let widthKey = [0,2,probMaxIdx] as [NSNumber]
                    let heightKey = [0,3,probMaxIdx] as [NSNumber]
                    maxProb = feature[nextKey].floatValue
                    box_width = feature[widthKey].floatValue
                    box_height = feature[heightKey].floatValue
                    
                    box_x = feature[xKey].floatValue - (box_width/2)
                    box_y = feature[yKey].floatValue - (box_height/2)
                }
            }
        }
        self.maxProbValue = "\(maxProb)"
        boundingBox = CGRect(x: CGFloat(box_x)
                             ,y: CGFloat(box_y)
                             ,width: CGFloat(box_width)
                             ,height: CGFloat(box_height))
        var maxMaskProb : Float = 0
        var maxMaskIdx = 0
        for maskPrbIdx in 5..<feature.shape[1].intValue-1{
            let key = [0,maskPrbIdx,probMaxIdx] as [NSNumber]
            let nextKey = [0,maskPrbIdx+1,probMaxIdx] as [NSNumber]
            if(feature[key].floatValue < feature[nextKey].floatValue){
                if(maxMaskProb < feature[nextKey].floatValue){
                    maxMaskIdx = maskPrbIdx+1
                    maxMaskProb = feature[nextKey].floatValue
                }
            }
            bestMaskIdx = maxMaskIdx-5
            print("\(maskPrbIdx-5) Best mask probablity is \(maxMaskIdx-5) with value \(maxMaskProb)")
        }
        
        print("Bounding box from classifier \(boundingBox)")
        return boundingBox
        
        
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
