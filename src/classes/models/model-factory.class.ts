import type { Classifier, Exercise } from "../../types";
import { EmpiricalModel } from "./empirical-model.class";
import { KerasModel, SklearnModel } from "./model.class";

export class ModelFactory {
  private static models: Record<Exercise, Record<string, Classifier>> = {
    high_plank: {
      Empírico: new EmpiricalModel(),
      FCNN: new KerasModel("/models/high-plank/fcnn/full_body_model.onnx", [
        "correct",
        "incorrect",
      ]),
      KNN: new SklearnModel("/models/high-plank/knn/full_body_model.onnx"),
      "Random Forest": new SklearnModel(
        "/models/high-plank/random-forest/full_body_model.onnx"
      ),
      "Regressão Logística": new SklearnModel(
        "/models/high-plank/logistic-regression/full_body_model.onnx"
      ),
      SVM: new SklearnModel("/models/high-plank/svm/full_body_model.onnx"),
    },
  };

  public static getExerciseModelNames(exercise: Exercise) {
    return Object.keys(this.models[exercise]).sort();
  }

  static getModel(exercise: Exercise, modelName: string): Classifier {
    return this.models[exercise][modelName];
  }
}
