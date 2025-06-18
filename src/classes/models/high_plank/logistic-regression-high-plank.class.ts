import type { Landmark } from "@mediapipe/tasks-vision";
import { anglesExtractor } from "../features-extractor.class";
import { LogisticRegressionModel } from "../logistic-regression";

export class LogisticRegressionHighPlankModel extends LogisticRegressionModel {
  modelPath = "models/high-plank/logistic-regression/full_body_model.json";

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
