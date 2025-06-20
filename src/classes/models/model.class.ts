import type { Landmark } from "@mediapipe/tasks-vision";
import type { LandmarkKey } from "../../types";
import { Tensor, tensor2d, type LayersModel } from "@tensorflow/tfjs";
import Point3d from "../point3d.class";
import Utils from "../utils.class";

export type ModelJson<P, M> = {
  params: P;
  features: { angles: LandmarkKey[][] };
  classes: string[];
  model_data: M;
};

export abstract class Model<P, M> {
  protected modelJson: ModelJson<P, M>;

  constructor(modelJson: ModelJson<P, M>) {
    this.modelJson = modelJson;
  }

  abstract predict(landmarks: Landmark[]): string;
}

type NeuralNetworkJson = {
  features: { angles: LandmarkKey[][] };
  classes: string[];
};
export class NeuralNetworkModel {
  private model: LayersModel;
  private modelJson: NeuralNetworkJson;

  constructor(model: LayersModel, modelJson: NeuralNetworkJson) {
    this.model = model;
    this.modelJson = modelJson;
  }

  predict(landmarks: Landmark[]): string | null {
    const x = this.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const inputTensor = tensor2d([x]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;
    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = this.modelJson.classes[predictedIndex];
    const translatedClass = Utils.translate(predictedClass);
    return `${translatedClass}(${maxProb.toFixed(2)})`;
  }
}
