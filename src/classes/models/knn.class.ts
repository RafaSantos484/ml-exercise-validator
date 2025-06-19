import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "./model.class";
import Utils from "../utils.class";
import type { LandmarkKey } from "../../types";

type WeightStrategy = "uniform" | "distance";

export type KnnJson = {
  params: {
    metric: "minkowski";
    n_neighbors: number;
    p: number;
    weights: WeightStrategy;
    train_test_split_seed: number;
  };
  classes: string[];
  features: { angles: LandmarkKey[][] };
  model_data: { X: number[][]; y: number[] };
};

export class KNNClassifier {
  modelJson: KnnJson;

  constructor(modelJson: KnnJson) {
    this.modelJson = modelJson;
  }

  private minkowskiDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must be of same length");
    }

    const { p } = this.modelJson.params;
    const sum = a.reduce(
      (acc, val, i) => acc + Math.pow(Math.abs(val - b[i]), p),
      0
    );
    return Math.pow(sum, 1 / p);
  }

  private getNeighbors(input: number[]): { label: number; distance: number }[] {
    const { X, y } = this.modelJson.model_data;
    const k = this.modelJson.params.n_neighbors;

    return X.map((xVec, i) => ({
      label: y[i],
      distance: this.minkowskiDistance(input, xVec),
    }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, k);
  }

  private vote(neighbors: { label: number; distance: number }[]): number {
    const votes: Record<number, number> = {};
    const { weights } = this.modelJson.params;

    const winner = { label: 0, votes: 0 };
    for (const { label, distance } of neighbors) {
      let weight = 1;
      if (weights === "distance") {
        weight = distance === 0 ? Infinity : 1 / distance;
      }

      if (!(label in votes)) {
        votes[label] = 0;
      }
      votes[label] += weight;
      if (votes[label] > winner.votes) {
        winner.label = label;
        winner.votes = votes[label];
      }
    }

    return winner.label;
  }

  predict(input: number[]): string {
    const neighbors = this.getNeighbors(input);
    const prediction = this.vote(neighbors);
    const label = this.modelJson.classes[prediction];
    return Utils.translate(label);
  }
}

export abstract class KnnModel implements Model {
  abstract modelPath: string;
  protected model: KNNClassifier | null = null;

  async load(): Promise<void> {
    if (!this.model) {
      const res = await fetch(this.modelPath);
      const modelJson: KnnJson = await res.json();
      this.model = new KNNClassifier(modelJson);
    }
  }

  abstract predict(landmarks: Landmark[]): string | null;
}
