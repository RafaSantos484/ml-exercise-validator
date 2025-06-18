import type { Landmark } from "@mediapipe/tasks-vision";
import type { Model } from "./model.class";
import type { LandmarkKey } from "../../types";

type TreeNode = {
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  value: number[][];
};

type Forest = TreeNode[];

export type RandomForestJson = {
  params: {
    n_estimators: number;
    criterion: "gini" | "entropy" | "log_loss";
    max_depth: null | number;
    min_samples_split: number;
    min_samples_leaf: number;
    bootstrap: boolean;
  };
  features: { points: LandmarkKey[] };
  classes: string[];
  forest: Forest;
};

class RandomForestClassifier {
  private forest: Forest;
  private classes: string[];

  constructor(forest: Forest, classes: string[]) {
    this.forest = forest;
    this.classes = classes;
  }

  private predictTree(tree: TreeNode, x: number[]): number[] {
    let node = 0;
    while (tree.children_left[node] !== -1) {
      if (x[tree.feature[node]] <= tree.threshold[node]) {
        node = tree.children_left[node];
      } else {
        node = tree.children_right[node];
      }
    }
    return tree.value[node];
  }

  predict(x: number[]): string {
    const numClasses = this.classes.length;
    const votes = new Array(numClasses).fill(0);

    this.forest.forEach((tree) => {
      const probs = this.predictTree(tree, x);
      const predicted = probs.indexOf(Math.max(...probs));
      votes[predicted]++;
    });

    const prediction = votes.indexOf(Math.max(...votes));
    const label = this.classes[prediction];
    const translator: Record<string, string> = {
      incorrect: "Incorreto",
      correct: "Correto",
    };
    return translator[label];
  }
}

export abstract class RandomForestModel implements Model {
  abstract modelPath: string;
  protected model: RandomForestClassifier | null = null;
  protected points: LandmarkKey[] = [];

  async load(): Promise<void> {
    if (!this.model) {
      const res = await fetch(this.modelPath);
      const modelDict: RandomForestJson = await res.json();
      const { forest, classes, features } = modelDict;
      this.points = features.points;
      this.model = new RandomForestClassifier(forest, classes);
    }
  }

  abstract predict(landmarks: Landmark[]): string | null;
}
