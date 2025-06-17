import type { Landmark } from "@mediapipe/tasks-vision";
import { KnnModel } from "../knn.class";
import { anglesExtractor } from "../features-extractor.class";

export class KnnHighPlankAnglesModel extends KnnModel {
  modelPath = "models/high-plank/knn/full_body_model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = anglesExtractor.getFeatures("high_plank", landmarks);
    return this.model.predict(angles);
  }
}
