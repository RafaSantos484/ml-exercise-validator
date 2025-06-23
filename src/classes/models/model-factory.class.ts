import type { Exercise } from "../../types";
import { NeuralNetworkModel, type Classifier } from "./model.class";
import { KnnModel } from "./knn.class";
import Utils from "../utils.class";
import { RandomForestModel } from "./random-forest.class";
import { LogisticRegressionModel } from "./logistic-regression";
import { loadLayersModel } from "@tensorflow/tfjs";
import { SvmModel } from "./svm.class";

export class ModelFactory {
  private static modelsGettersPerExercise: Record<
    Exercise,
    Record<string, () => Promise<Classifier>>
  > = {
    high_plank: {
      FCNN: async () =>
        new NeuralNetworkModel(
          await loadLayersModel("models/high-plank/fcnn/model.json"),
          await Utils.readJson(
            "models/high-plank/fcnn/full_body_model_info.json"
          )
        ),
      KNN: async () =>
        new KnnModel(
          await Utils.readJson("models/high-plank/knn/full_body_model.json")
        ),
      "Random Forest": async () =>
        new RandomForestModel(
          await Utils.readJson(
            "models/high-plank/random-forest/full_body_model.json"
          )
        ),
      "Regressão Logística": async () =>
        new LogisticRegressionModel(
          await Utils.readJson(
            "models/high-plank/logistic-regression/full_body_model.json"
          )
        ),
      SVM: async () =>
        new SvmModel(
          await Utils.readJson("models/high-plank/svm/full_body_model.json")
        ),
    },
  };

  private static models: Record<Exercise, Record<string, Classifier>> = {
    high_plank: {},
  };

  public static getExerciseModelNames(exercise: Exercise) {
    return Object.keys(this.modelsGettersPerExercise[exercise]).sort();
  }

  static async getModel(
    exercise: Exercise,
    modelName: string
  ): Promise<Classifier> {
    if (!this.models[exercise][modelName]) {
      this.models[exercise][modelName] = await this.modelsGettersPerExercise[
        exercise
      ][modelName]();
    }
    return this.models[exercise][modelName];
  }
}
