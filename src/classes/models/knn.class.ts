import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "./model.class";

type WeightStrategy = "uniform" | "distance";

export type KnnJson = {
  params: {
    metric: "minkowski";
    n_neighbors: number;
    p: number;
    weights: WeightStrategy;
  };
  classes: string[];
  X: number[][];
  y: number[];
};

export class KNNClassifier {
  private X: number[][] = [];
  private y: number[] = [];
  private classes: string[];
  private k: number;
  private p: number;
  private weights: WeightStrategy;

  constructor(
    classes: string[],
    k: number,
    p: number,
    weights: WeightStrategy
  ) {
    this.classes = classes;
    this.k = k;
    this.p = p;
    this.weights = weights;
  }

  fit(X: number[][], y: number[]) {
    if (X.length !== y.length) {
      throw new Error("X and y must have the same length");
    }

    this.X = X;
    this.y = y;
  }

  private minkowskiDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must be of same length");
    }

    const sum = a.reduce(
      (acc, val, i) => acc + Math.pow(Math.abs(val - b[i]), this.p),
      0
    );
    return Math.pow(sum, 1 / this.p);
  }

  private getNeighbors(input: number[]): { label: number; distance: number }[] {
    return this.X.map((xVec, i) => ({
      label: this.y[i],
      distance: this.minkowskiDistance(input, xVec),
    }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, this.k);
  }

  private vote(neighbors: { label: number; distance: number }[]): number {
    const votes: Record<number, number> = {};

    const winner = { label: 0, votes: 0 };
    for (const { label, distance } of neighbors) {
      let weight = 1;
      if (this.weights === "distance") {
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
    const label = this.classes[prediction];
    const translator: Record<string, string> = {
      incorrect: "Incorreto",
      correct: "Correto",
    };
    return translator[label];
  }
}

export abstract class KnnModel implements Model {
  abstract modelPath: string;
  protected model: KNNClassifier | null = null;

  async load(): Promise<void> {
    if (!this.model) {
      const res = await fetch(this.modelPath);
      const modelDict: KnnJson = await res.json();
      const { n_neighbors, p, weights } = modelDict.params;
      this.model = new KNNClassifier(
        modelDict.classes,
        n_neighbors,
        p,
        weights
      );
      this.model.fit(modelDict.X, modelDict.y);
    }
  }

  abstract predict(landmarks: Landmark[]): string | null;
}
