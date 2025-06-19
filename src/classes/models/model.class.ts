import type { Landmark } from "@mediapipe/tasks-vision";
import type { LandmarkKey } from "../../types";

export type ClassifierJson<P, M> = {
  params: P;
  features: { angles: LandmarkKey[][] };
  classes: string[];
  model_data: M;
};

/*
export abstract class Classifier<P, M> {
  modelJson: ClassifierJson<P, M>;

  constructor(modelJson: ClassifierJson<P, M>) {
    this.modelJson = modelJson;
  }

  abstract predict(x: number[]): string;
}
*/

export abstract class Model<P, M> {
  protected modelJson: ClassifierJson<P, M>;

  constructor(modelJson: ClassifierJson<P, M>) {
    this.modelJson = modelJson;
  }

  abstract predict(landmarks: Landmark[]): string | null;
}
