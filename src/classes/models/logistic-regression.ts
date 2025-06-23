import type { Landmark } from "@mediapipe/tasks-vision";
import { NonNeuralModel } from "./model.class";
import Utils from "../utils.class";
import Point3d from "../point3d.class";

type LogisticRegressionParams = {
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
type LogisticRegressionModelData = { coef: number[][]; intercept: number[] };

export class LogisticRegressionModel extends NonNeuralModel<
  LogisticRegressionParams,
  LogisticRegressionModelData
> {
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

  predict(landmarks: Landmark[]): string {
    const x = this.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const [prediction, prob] = this.logisticPredict(x);
    const label = this.modelJson.classes[prediction];
    const translatedLabel = Utils.translate(label);
    return `${translatedLabel}(${prob.toFixed(2)})`;
  }
}
