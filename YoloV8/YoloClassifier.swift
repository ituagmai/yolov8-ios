//
//  YoloClassifier.swift
//  YoloV8
//
//

import Foundation
import CoreML
import Vision
class YoloClassifier{
    
    
    
    func classifyImage()-> bestOutput?{
        
        let bestModelWrapper = try? best()
        
        guard let bestModel = bestModelWrapper else{
            return nil
        }
        let imageUrl = Bundle.main.url(forResource: "tomcruise", withExtension: "jpeg")!
        do{
            let output = try bestModel.prediction(input: bestInput(imageAt: imageUrl))
            print(output)
            return output
        }
        catch{
            print("\(error)")
        }
       return nil
    }
    
    func classifyVisionModel(onClassified: @escaping ([Any])->Void){
        let bestModelWrapper = try? best()
        
        guard let bestModel = bestModelWrapper else{
            return
        }
        do
        {
            let visionModel = try VNCoreMLModel(for: bestModel.model)
            let segmentationRequest = VNCoreMLRequest(model: visionModel, completionHandler: {(req,err) in
                if let results = req.results{
                    onClassified(results)
                }
            })
            let processingRequests = [segmentationRequest]
            let segmentationRequestHandler = VNImageRequestHandler(url: Bundle.main.url(forResource: "tomcruise", withExtension: "jpeg")!
                                                                   , orientation: .up)
            try segmentationRequestHandler.perform(processingRequests)
        }
        catch{
            print("\(error)")
        }
    }
}
