import type { Landmark } from "@mediapipe/tasks-vision";
import type { Constructor, Exercise } from "../../types";
import {
  FCNNHighPlankAnglesFullBodyModel,
  FCNNHighPlankPointsFullBodyModel,
} from "./fcnn.class";

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
      "FCNN - Ângulos": FCNNHighPlankAnglesFullBodyModel,
      "FCNN - Pontos": FCNNHighPlankPointsFullBodyModel,
    },
  };

  private static models: Record<string, Record<string, Model>> = {};

  public static getExerciseModelNames(exercise: Exercise) {
    return Object.keys(this.modelsPerExercise[exercise]);
  }

  public static getModel(exercise: Exercise, modelName: string) {
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
}
