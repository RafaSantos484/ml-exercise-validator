import { Tensor, tensor2d } from "@tensorflow/tfjs";
import type { Landmark } from "@mediapipe/tasks-vision";
import { NeuralNetworkModel } from "../neural-network.class";
import { anglesExtractor } from "../features-extractor.class";

export class FcnnHighPlankAnglesModel extends NeuralNetworkModel {
  modelPath = "models/high-plank/fcnn/model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = anglesExtractor.getFeatures("high_plank", landmarks);
    const inputTensor = tensor2d([angles]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;
    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}
