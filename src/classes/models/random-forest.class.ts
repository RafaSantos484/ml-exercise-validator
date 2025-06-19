import { Model } from "./model.class";
import Utils from "../utils.class";
import Point3d from "../point3d.class";
import type { Landmark } from "@mediapipe/tasks-vision";

type TreeNode = {
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  value: number[][];
};
type Forest = TreeNode[];

type RandomForestParams = {
  n_estimators: number;
  criterion: "gini" | "entropy" | "log_loss";
  max_depth: null | number;
  min_samples_split: number;
  min_samples_leaf: number;
  bootstrap: boolean;
};
type RandomForestModelData = Forest;

export class RandomForestModel extends Model<
  RandomForestParams,
  RandomForestModelData
> {
  private static predictTree(tree: TreeNode, x: number[]): number[] {
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

  predict(landmarks: Landmark[]): string {
    const x = this.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const forest = this.modelJson.model_data;
    const { classes } = this.modelJson;
    const numClasses = classes.length;
    const votes = new Array(numClasses).fill(0);

    forest.forEach((tree) => {
      const probs = RandomForestModel.predictTree(tree, x);
      const predicted = probs.indexOf(Math.max(...probs));
      votes[predicted]++;
    });

    const prediction = votes.indexOf(Math.max(...votes));
    const label = classes[prediction];
    return Utils.translate(label);
  }
}
