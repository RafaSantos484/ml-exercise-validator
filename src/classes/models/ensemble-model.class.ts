import type { Landmark } from "@mediapipe/tasks-vision";
import type { Classifier, ValidationResult } from "../../types";
import Utils from "../utils.class";

export class EnsembleModel implements Classifier {
  private models: Classifier[];

  constructor(...models: Classifier[]) {
    this.models = models;
  }

  async load() {
    const loadPromises = this.models.map((model) => model.load());
    await Promise.all(loadPromises);
  }

  async predict(landmarks: Landmark[]): Promise<ValidationResult> {
    // Running all models in parallel may cause the error "Error: Session already started"
    /*
    const results = await Promise.all(
      this.models.map((model) => model.predict(landmarks))
    );
    */
    const results: ValidationResult[] = [];
    for (const model of this.models) {
      const result = await model.predict(landmarks);
      results.push(result);
    }
    let isCorrectSum = 0;
    let isIncorrectSum = 0;
    results.forEach((result) => {
      if (result.isCorrect) {
        isCorrectSum++;
      } else {
        isIncorrectSum++;
      }
    });
    const isCorrect = isCorrectSum > isIncorrectSum;
    return Utils.translate(isCorrect ? "correct" : "incorrect");
  }
}
