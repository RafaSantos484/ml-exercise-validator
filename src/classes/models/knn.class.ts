import { NonNeuralModel } from "./model.class";
import Utils from "../utils.class";
import type { Landmark } from "@mediapipe/tasks-vision";
import Point3d from "../point3d.class";

type KnnParams = {
  metric: "minkowski";
  n_neighbors: number;
  p: number;
  weights: "uniform" | "distance";
  train_test_split_seed: number;
};
type KnnModelData = { X: number[][]; y: number[] };
// type KnnJson = ClassifierJson<KnnParams, KnnModelData>;

export class KnnModel extends NonNeuralModel<KnnParams, KnnModelData> {
  private getNeighbors(input: number[]): { label: number; distance: number }[] {
    const { X, y } = this.modelJson.model_data;
    const k = this.modelJson.params.n_neighbors;

    return X.map((xVec, i) => ({
      label: y[i],
      distance: Utils.minkowskiDistance(input, xVec, this.modelJson.params.p),
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

  predict(landmarks: Landmark[]): string {
    const x = this.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const neighbors = this.getNeighbors(x);
    const prediction = this.vote(neighbors);
    const label = this.modelJson.classes[prediction];
    return Utils.translate(label);
  }
}
