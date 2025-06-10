import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "../models/model.class";

export abstract class Validator {
  protected model: Model;

  constructor(model: Model) {
    this.model = model;
  }

  async load() {
    await this.model.load();
  }

  abstract validate(landmarks: Landmark[]): [string, number] | null;
}
