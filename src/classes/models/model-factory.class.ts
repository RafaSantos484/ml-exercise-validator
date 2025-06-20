import type { Landmark } from "@mediapipe/tasks-vision";
import type { Exercise } from "../../types";
import { NeuralNetworkModel, type Model } from "./model.class";
import { KnnModel } from "./knn.class";
import Utils from "../utils.class";
import { RandomForestModel } from "./random-forest.class";
import { LogisticRegressionModel } from "./logistic-regression";
import { loadLayersModel } from "@tensorflow/tfjs";
import { SvmModel } from "./svm.class";

type GenericModel = Model<any, any> | NeuralNetworkModel;

export class ModelFactory {
  private static modelsGettersPerExercise: Record<
    Exercise,
    Record<string, () => Promise<GenericModel>>
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

  private static models: Record<Exercise, Record<string, GenericModel>> = {
    high_plank: {},
  };

  public static getExerciseModelNames(exercise: Exercise) {
    return Object.keys(this.modelsGettersPerExercise[exercise]).sort();
  }

  private static getModel(
    exercise: Exercise,
    modelName: string
  ): GenericModel | undefined {
    return this.models[exercise][modelName];
  }

  static async loadModel(exercise: Exercise, modelName: string) {
    const model: GenericModel | undefined = this.models[exercise][modelName];
    if (!model) {
      this.models[exercise][modelName] = await this.modelsGettersPerExercise[
        exercise
      ][modelName]();
    }
  }

  static predict(exercise: Exercise, modelName: string, landmarks: Landmark[]) {
    const model = this.getModel(exercise, modelName);
    if (!model) {
      return null;
    } else {
      return model.predict(landmarks);
    }
  }
}
