import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "./model.class";
import type { LandmarkKey } from "../../types";
import Utils from "../utils.class";

export type LogisticRegressionJson = {
  params: {
    C: number;
    dual: boolean;
    fit_intercept: boolean;
    max_iter: number;
    penalty: null | "l1" | "l2" | "elasticnet";
    solver:
      | "lbfgs"
      | "liblinear"
      | "newton-cg"
      | "newton-cholesky"
      | "sag"
      | "saga";
  };
  features: { angles: LandmarkKey[][] };
  classes: string[];
  model_data: { coef: number[][]; intercept: number[] };
};

class LogisticRegressionClassifier {
  modelJson: LogisticRegressionJson;

  constructor(modelJson: LogisticRegressionJson) {
    this.modelJson = modelJson;
  }

  private logisticPredict(x: number[]): [number, number] {
    const { coef, intercept } = this.modelJson.model_data;
    const logits = coef.map((weights, i) =>
      weights.reduce((sum, wj, j) => sum + wj * x[j], intercept[i])
    );
    const probs =
      coef.length === 1
        ? [1 - Utils.sigmoid(logits[0]), Utils.sigmoid(logits[0])]
        : Utils.softmax(logits);

    const maxIdx = probs.indexOf(Math.max(...probs));
    return [maxIdx, probs[maxIdx]];
  }

  predict(x: number[]): string {
    const [prediction, prob] = this.logisticPredict(x);
    const label = this.modelJson.classes[prediction];
    const translatedLabel = Utils.translate(label);
    return `${translatedLabel}(${prob.toFixed(2)})`;
  }
}

export abstract class LogisticRegressionModel implements Model {
  abstract modelPath: string;
  protected model: LogisticRegressionClassifier | null = null;
  protected points: LandmarkKey[] = [];

  async load(): Promise<void> {
    if (!this.model) {
      const res = await fetch(this.modelPath);
      const modelJson: LogisticRegressionJson = await res.json();
      this.model = new LogisticRegressionClassifier(modelJson);
    }
  }

  abstract predict(landmarks: Landmark[]): string | null;
}
