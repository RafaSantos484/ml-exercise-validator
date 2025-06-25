import type { Exercise } from "../../types";
import { type Classifier } from "./model.class";
import { KnnModel } from "./knn.class";
import { RandomForestModel } from "./random-forest.class";
import { SvmModel } from "./svm.class";
import { LogisticRegressionModel } from "./logistic-regression";

export class ModelFactory {
  private static models: Record<Exercise, Record<string, Classifier>> = {
    high_plank: {
      KNN: new KnnModel(),
      "Random Forest": new RandomForestModel(),
      "Regressão Logística": new LogisticRegressionModel(),
      SVM: new SvmModel(),
    },
  };

  public static getExerciseModelNames(exercise: Exercise) {
    return Object.keys(this.models[exercise]).sort();
  }

  static getModel(exercise: Exercise, modelName: string): Classifier {
    return this.models[exercise][modelName];
  }
}
