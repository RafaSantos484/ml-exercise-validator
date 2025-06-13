import type { Landmark } from "@mediapipe/tasks-vision";
import type { Constructor, Exercise } from "../../types";
import { KnnHighPlankAnglesModel } from "./high_plank/knn-high-plank.class";
import {
  CnnHighPlankPointsModel,
  FcnnHighPlankAnglesModel,
  FcnnHighPlankPointsModel,
} from "./high_plank/neural-network-high-plank.class";

export abstract class Model {
  abstract load(): Promise<void>;
  abstract predict(landmarks: Landmark[]): string | null;
}

type ModelChild = Constructor<Model>;

export class ModelFactory {
  private static modelsPerExercise: Record<
    Exercise,
    Record<string, ModelChild>
  > = {
    high_plank: {
      "FCNN - Ângulos": FcnnHighPlankAnglesModel,
      "FCNN - Pontos": FcnnHighPlankPointsModel,
      "CNN - Pontos": CnnHighPlankPointsModel,
      "KNN - Ângulos": KnnHighPlankAnglesModel,
    },
  };

  private static models: Record<string, Record<string, Model>> = {};

  public static getExerciseModelNames(exercise: Exercise) {
    return Object.keys(this.modelsPerExercise[exercise]);
  }

  private static getModel(exercise: Exercise, modelName: string) {
    if (!this.models[exercise]) {
      this.models[exercise] = {};
    }
    if (!this.models[exercise][modelName]) {
      this.models[exercise][modelName] = new this.modelsPerExercise[exercise][
        modelName
      ]();
    }

    return this.models[exercise][modelName];
  }

  static async loadModel(exercise: Exercise, modelName: string) {
    await this.getModel(exercise, modelName).load();
  }

  static predict(exercise: Exercise, modelName: string, landmarks: Landmark[]) {
    const model = this.getModel(exercise, modelName);
    return model.predict(landmarks);
  }
}
