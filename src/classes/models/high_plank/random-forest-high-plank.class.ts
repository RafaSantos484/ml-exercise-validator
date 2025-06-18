import type { Landmark } from "@mediapipe/tasks-vision";
import { anglesExtractor } from "../features-extractor.class";
import { RandomForestModel } from "../random-forest.class";

export class RandomForestHighPlankAnglesModel extends RandomForestModel {
  modelPath = "models/high-plank/random-forest/full_body_model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = anglesExtractor.getAnglesCombinations(
      landmarks,
      this.points
    );
    return this.model.predict(angles);
  }
}
