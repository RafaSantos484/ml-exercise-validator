import type { Landmark } from "@mediapipe/tasks-vision";
import type { LandmarkKey } from "../../types";
import { tensor2d, type LayersModel } from "@tensorflow/tfjs";
import Point3d from "../point3d.class";
import Utils from "../utils.class";
import { InferenceSession, Tensor, env } from "onnxruntime-web";

env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

export type ModelJson<P, M> = {
  params: P;
  features: { angles: LandmarkKey[][] };
  classes: string[];
  model_data: M;
};

export interface Classifier {
  predict(landmarks: Landmark[]): Promise<string>;
  load(): Promise<void>;
}

export abstract class Model implements Classifier {
  // protected modelJson: ModelJson<P, M>;
  protected abstract modelPath: string;
  private session: InferenceSession | undefined;

  async load(): Promise<void> {
    if (!this.session) {
      this.session = await InferenceSession.create(this.modelPath);
    }
  }

  protected async getSession() {
    await this.load();
    return this.session as InferenceSession;
  }

  getTensor(landmarks: Landmark[]) {
    const triplets: LandmarkKey[][] = [
      ["LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER"],
      ["LEFT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "LEFT_SHOULDER"],
      ["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_HIP"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP"],
      ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
      ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
      ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
      ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
      ["LEFT_ANKLE", "LEFT_HIP", "RIGHT_HIP"],
      ["RIGHT_ANKLE", "RIGHT_HIP", "LEFT_HIP"],
      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "LEFT_SHOULDER"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "RIGHT_SHOULDER"],
      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "RIGHT_WRIST"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "LEFT_WRIST"],
    ];
    const angles = triplets.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const data = Float32Array.from(angles);
    const dims = [1, angles.length];
    const tensor = new Tensor("float32", data, dims);
    return tensor;
  }

  abstract predict(landmarks: Landmark[]): Promise<string>;
}

export class NeuralNetworkModel extends Model<undefined, undefined> {
  private model: LayersModel;

  constructor(model: LayersModel, modelJson: ModelJson<undefined, undefined>) {
    super(modelJson);
    this.model = model;
  }

  async predict(landmarks: Landmark[]): Promise<string> {
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
