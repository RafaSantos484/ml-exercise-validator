import type { Landmark } from "@mediapipe/tasks-vision";
import { EmpiricalModel } from "./empirical-model.class";
import { SklearnModel } from "./sklearn-model.class";
import type { ValidationResult } from "../../types";

export class HybridModel extends SklearnModel {
  private empiricalModel: EmpiricalModel;

  constructor(
    modelPath: string,
    empiricalModel: EmpiricalModel = new EmpiricalModel()
  ) {
    super(modelPath);
    this.empiricalModel = empiricalModel;
  }

  async load() {
    await super.load();
    // await this.empiricalModel.load();
  }

  async predict(landmarks: Landmark[]): Promise<ValidationResult> {
    const sklearnResult = await super.predict(landmarks);
    if (sklearnResult.isCorrect) {
      return sklearnResult;
    }

    const empiricalResult = await this.empiricalModel.predict(landmarks);
    return empiricalResult.isCorrect ? empiricalResult : sklearnResult;
  }
}
