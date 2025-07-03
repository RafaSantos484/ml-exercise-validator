import type { Landmark } from "@mediapipe/tasks-vision";
import { EmpiricalModel } from "./empirical-model.class";
import { SklearnModel } from "./sklearn-model.class";
import type { Classifier, ValidationResult } from "../../types";

export class HybridModel implements Classifier {
  private sklearnModel: SklearnModel;
  private empiricalModel: EmpiricalModel;

  constructor(
    sklearnModel: SklearnModel,
    empiricalModel: EmpiricalModel = new EmpiricalModel()
  ) {
    this.sklearnModel = sklearnModel;
    this.empiricalModel = empiricalModel;
  }

  async load() {
    await this.sklearnModel.load();
    // await this.empiricalModel.load();
  }

  async predict(landmarks: Landmark[]): Promise<ValidationResult> {
    const sklearnResult = await this.sklearnModel.predict(landmarks);
    if (sklearnResult.isCorrect) {
      return sklearnResult;
    }

    const empiricalResult = await this.empiricalModel.predict(landmarks);
    return empiricalResult.isCorrect ? sklearnResult : empiricalResult;
  }
}
