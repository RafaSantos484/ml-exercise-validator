import type { Landmark } from "@mediapipe/tasks-vision";
import { NonNeuralModel } from "./model.class";
import Point3d from "../point3d.class";
import Utils from "../utils.class";

type KernelType = "linear" | "poly" | "rbf" | "sigmoid";
type SvmParams = {
  C: number;
  decision_function_shape: "ovo" | "ovr";
  gamma: "scale" | "auto";
  kernel: KernelType;
  probability: boolean;
  shrinking: boolean;
  train_test_split_seed: number;
};
type SvmModelData = {
  kernel: KernelType;
  support_vectors: number[][];
  dual_coef: number[][];
  intercept: number[];
  gamma: number;
  coef0: number;
  degree: number;
  n_support: number[];
};

export class SvmModel extends NonNeuralModel<SvmParams, SvmModelData> {
  private static linearKernel(x: number[], y: number[]): number {
    return x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  }

  private static polyKernel(
    x: number[],
    y: number[],
    degree: number,
    coef0: number,
    gamma: number
  ): number {
    return Math.pow(gamma * this.linearKernel(x, y) + coef0, degree);
  }

  private static rbfKernel(x: number[], y: number[], gamma: number): number {
    const squaredDistance = x.reduce((sum, xi, i) => {
      const diff = xi - y[i];
      return sum + diff * diff;
    }, 0);
    return Math.exp(-gamma * squaredDistance);
  }

  private static sigmoidKernel(
    x: number[],
    y: number[],
    gamma: number,
    coef0: number
  ): number {
    return Math.tanh(gamma * this.linearKernel(x, y) + coef0);
  }

  private static computeKernel(
    x: number[],
    y: number[],
    model: SvmModelData
  ): number {
    const { kernel, gamma, coef0, degree } = model;

    switch (kernel) {
      case "linear":
        return this.linearKernel(x, y);
      case "poly":
        return this.polyKernel(x, y, degree, coef0, gamma);
      case "rbf":
        return this.rbfKernel(x, y, gamma);
      case "sigmoid":
        return this.sigmoidKernel(x, y, gamma, coef0);
      default:
        throw new Error(`Unsupported kernel type: ${kernel}`);
    }
  }

  predict(landmarks: Landmark[]): string {
    const input = this.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const { support_vectors, dual_coef, intercept, n_support } =
      this.modelJson.model_data;
    const { decision_function_shape } = this.modelJson.params;

    const classLabels = this.modelJson.classes;
    const nClasses = classLabels.length;
    let resultLabel = "";

    // Binary classification (same logic in both ovo/ovr)
    if (nClasses === 2) {
      let decision = 0;
      for (let i = 0; i < dual_coef[0].length; i++) {
        decision +=
          dual_coef[0][i] *
          SvmModel.computeKernel(
            input,
            support_vectors[i],
            this.modelJson.model_data
          );
      }
      decision += intercept[0];
      resultLabel = decision > 0 ? classLabels[1] : classLabels[0];
    }
    // Multiclass OVO (One-vs-One)
    else if (decision_function_shape === "ovo") {
      const votes = new Array(nClasses).fill(0);
      let svIndex = 0;
      let pairIndex = 0;

      for (let i = 0; i < nClasses; i++) {
        for (let j = i + 1; j < nClasses; j++) {
          const coef = dual_coef[pairIndex];
          const intercept_ij = intercept[pairIndex];

          const n_i = n_support[i];
          const n_j = n_support[j];
          const svs_ij = support_vectors.slice(svIndex, svIndex + n_i + n_j);

          let decision = 0;
          for (let k = 0; k < coef.length; k++) {
            decision +=
              coef[k] *
              SvmModel.computeKernel(
                input,
                svs_ij[k],
                this.modelJson.model_data
              );
          }
          decision += intercept_ij;

          if (decision > 0) {
            votes[j]++;
          } else {
            votes[i]++;
          }

          pairIndex++;
          svIndex += n_i + n_j;
        }
      }

      const predictedIndex = votes.indexOf(Math.max(...votes));
      resultLabel = classLabels[predictedIndex];
    }
    // Multiclass OVR (One-vs-Rest)
    else if (decision_function_shape === "ovr") {
      const decisions: number[] = [];

      let svIndex = 0;
      for (let i = 0; i < nClasses; i++) {
        const coef = dual_coef[i];
        const intercept_i = intercept[i];
        const n_i = n_support[i];
        const svs_i = support_vectors.slice(svIndex, svIndex + n_i);

        let decision = 0;
        for (let k = 0; k < coef.length; k++) {
          decision +=
            coef[k] *
            SvmModel.computeKernel(input, svs_i[k], this.modelJson.model_data);
        }
        decision += intercept_i;
        decisions.push(decision);

        svIndex += n_i;
      }

      const predictedIndex = decisions.indexOf(Math.max(...decisions));
      return classLabels[predictedIndex];
    } else {
      throw new Error(
        `Unsupported decision_function_shape: ${decision_function_shape}`
      );
    }

    return Utils.translate(resultLabel);
  }
}
