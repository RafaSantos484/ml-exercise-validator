import type { Landmark } from "@mediapipe/tasks-vision";
import { KnnModel } from "../knn.class";
import { FeaturesExtractor } from "../../features-extractor.class";

export class KnnHighPlankAnglesModel extends KnnModel {
  modelPath = "models/high-plank/knn-angles/full_body_model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = FeaturesExtractor.getAnglesFeatures(landmarks);
    return this.model.predict(angles);
  }
}
