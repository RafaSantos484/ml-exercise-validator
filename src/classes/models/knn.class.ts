import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "./model.class";
import type { LandmarkKey } from "../../types";
import Point3d from "../point3d.class";

type WeightStrategy = "uniform" | "distance";

type KnnJson = {
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

export class KnnHighPlankAnglesFullBodyModel implements Model {
  private model: KNNClassifier | null = null;

  async load(): Promise<void> {
    if (!this.model) {
      const res = await fetch(
        "src/assets/models/knn-high-plank-angles/full_body_model.json"
      );
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

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const triplets: [LandmarkKey, LandmarkKey, LandmarkKey][] = [
      ["LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER"],
      ["LEFT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
      ["LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_ELBOW"],
      ["LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
      ["RIGHT_ELBOW", "RIGHT_SHOULDER", "LEFT_SHOULDER"],
      ["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_HIP"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP"],

      ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
      ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
      ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
      ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
      ["LEFT_ANKLE", "LEFT_HIP", "RIGHT_HIP"],
      ["RIGHT_ANKLE", "RIGHT_HIP", "LEFT_HIP"],
      ["LEFT_ANKLE", "LEFT_KNEE", "RIGHT_KNEE"],
      ["RIGHT_ANKLE", "RIGHT_KNEE", "LEFT_KNEE"],

      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "LEFT_ELBOW"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "RIGHT_ELBOW"],
      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "LEFT_SHOULDER"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "RIGHT_SHOULDER"],
      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "RIGHT_WRIST"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "LEFT_WRIST"],
    ];
    const features = triplets.map((tp) =>
      Point3d.get_angle_from_joints_triplet(landmarks, tp)
    );
    return this.model.predict(features);
  }
}
