import { Tensor, tensor2d, tensor3d } from "@tensorflow/tfjs";
import type { Landmark } from "@mediapipe/tasks-vision";
import { NeuralNetworkModel } from "../neural-network.class";
import { FeaturesExtractor } from "../../features-extractor.class";

export class CnnHighPlankPointsModel extends NeuralNetworkModel {
  modelPath = "models/high-plank/cnn-points/full-body-model/model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const points = FeaturesExtractor.getPointsFeatures(landmarks);
    const inputTensor = tensor3d([points]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;
    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}

export class FcnnHighPlankAnglesModel extends NeuralNetworkModel {
  modelPath = "models/high-plank/fcnn-angles/full-body-model/model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = FeaturesExtractor.getAnglesFeatures(landmarks);
    const inputTensor = tensor2d([angles]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;
    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}

export class NnHighPlankMergedModel extends NeuralNetworkModel {
  modelPath = "models/high-plank/nn-merged/full-body-model/model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = FeaturesExtractor.getAnglesFeatures(landmarks);
    const points = FeaturesExtractor.getPointsFeatures(landmarks);
    const anglesTensor = tensor2d([angles]);
    const pointsTensor = tensor3d([points]);
    const outputTensor = this.model.predict([
      anglesTensor,
      pointsTensor,
    ]) as Tensor;

    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}
