import {
  loadLayersModel,
  regularizers,
  serialization,
  type LayersModel,
} from "@tensorflow/tfjs";
import type { Model } from "./model.class";
import type { Landmark } from "@mediapipe/tasks-vision";
import type { L1L2Args } from "@tensorflow/tfjs-layers/dist/regularizers";

class L2 {
  static className = "L2";

  constructor(config: L1L2Args) {
    return regularizers.l1l2(config);
  }
}
serialization.registerClass(L2 as any);

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
