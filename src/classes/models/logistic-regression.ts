import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "./model.class";
import type { LandmarkKey } from "../../types";

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
  features: { points: LandmarkKey[] };
  classes: string[];
  model_data: { coef: number[][]; intercept: number[] };
};

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits); // evita overflow
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

class RandomForestClassifier {
  private coef: number[][];
  private intercept: number[];
  private classes: string[];

  constructor(coef: number[][], intercept: number[], classes: string[]) {
    this.coef = coef;
    this.intercept = intercept;
    this.classes = classes;
  }

  private logisticPredict(x: number[]): [number, number] {
    const logits = this.coef.map((weights, i) =>
      weights.reduce((sum, wj, j) => sum + wj * x[j], this.intercept[i])
    );
    const probs =
      this.coef.length === 1
        ? [1 - sigmoid(logits[0]), sigmoid(logits[0])]
        : softmax(logits);

    const maxIdx = probs.indexOf(Math.max(...probs));
    return [maxIdx, probs[maxIdx]];
  }

  predict(x: number[]): string {
    const [prediction, prob] = this.logisticPredict(x);
    const label = this.classes[prediction];
    const translator: Record<string, string> = {
      incorrect: "Incorreto",
      correct: "Correto",
    };
    return `${translator[label]}(${prob.toFixed(2)})`;
  }
}

export abstract class LogisticRegressionModel implements Model {
  abstract modelPath: string;
  protected model: RandomForestClassifier | null = null;
  protected points: LandmarkKey[] = [];

  async load(): Promise<void> {
    if (!this.model) {
      const res = await fetch(this.modelPath);
      const modelDict: LogisticRegressionJson = await res.json();
      const { classes, features, model_data } = modelDict;
      this.points = features.points;
      this.model = new RandomForestClassifier(
        model_data.coef,
        model_data.intercept,
        classes
      );
    }
  }

  abstract predict(landmarks: Landmark[]): string | null;
}
