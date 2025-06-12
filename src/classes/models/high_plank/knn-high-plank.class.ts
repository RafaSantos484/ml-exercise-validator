import type { Landmark } from "@mediapipe/tasks-vision";
import Point3d from "../../point3d.class";
import type { LandmarkKey } from "../../../types";
import { KnnModel } from "../knn.class";

export class KnnHighPlankAnglesModel extends KnnModel {
  modelPath = "models/high-plank/knn-angles/full_body_model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const triplets: [LandmarkKey, LandmarkKey, LandmarkKey][] = [
      ["LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER"],
      ["LEFT_WRIST", "LEFT_ELBOW", "RIGHT_ELBOW"],
      ["RIGHT_WRIST", "RIGHT_ELBOW", "LEFT_ELBOW"],
      ["LEFT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "LEFT_SHOULDER"],
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
