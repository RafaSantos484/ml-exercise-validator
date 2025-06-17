import type { Landmark } from "@mediapipe/tasks-vision";
import { type Exercise, type LandmarkKey } from "../../types";
import Point3d from "../point3d.class";
import Utils from "../utils.class";

export class anglesExtractor {
  private static pointsPerExercise: Record<Exercise, LandmarkKey[]> = {
    high_plank: [
      "LEFT_WRIST",
      "RIGHT_WRIST",
      "LEFT_ELBOW",
      "RIGHT_ELBOW",
      "LEFT_SHOULDER",
      "RIGHT_SHOULDER",
      "LEFT_HIP",
      "RIGHT_HIP",
      "LEFT_KNEE",
      "RIGHT_KNEE",
      "LEFT_ANKLE",
      "RIGHT_ANKLE",
    ],
  };

  private static getAnglesCombinations(
    landmarks: Landmark[],
    points: LandmarkKey[]
  ) {
    const triplets = Utils.getCombinations(points, 3);
    const angles = triplets.map((tp) =>
      Point3d.getAngleFromJointsTriplet(landmarks, tp)
    );
    return angles;
  }

  static getFeatures(exercise: Exercise, landmarks: Landmark[]) {
    return this.getAnglesCombinations(
      landmarks,
      this.pointsPerExercise[exercise]
    );
  }
}
