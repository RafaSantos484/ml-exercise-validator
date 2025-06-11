import { loadLayersModel, type LayersModel } from "@tensorflow/tfjs";
import type { Model } from "./model.class";
import type { Landmark } from "@mediapipe/tasks-vision";

export abstract class NeuralNetworkModel implements Model {
  abstract modelPath: string;
  protected model: LayersModel | null = null;

  public async load() {
    if (!this.model) {
      this.model = await loadLayersModel(this.modelPath);
    }
  }
  abstract predict(landmarks: Landmark[]): string | null;
}
