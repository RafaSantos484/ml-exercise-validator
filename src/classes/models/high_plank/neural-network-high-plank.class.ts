import { tensor, Tensor } from "@tensorflow/tfjs";
import type { Landmark } from "@mediapipe/tasks-vision";
import Point3d from "../../point3d.class";
import { landmarksDict, type LandmarkKey } from "../../../types";
import { NeuralNetworkModel } from "../neural-network.class";
import CoordinateSystem3D from "../../coordinate-system-3d.class";

export class CnnHighPlankPointsModel extends NeuralNetworkModel {
  modelPath = "models/high-plank/cnn-points/full-body-model/model.json";

  private getCustomBasis(landmarks: Landmark[]) {
    const left_wrist_point = new Point3d(
      landmarks[landmarksDict["LEFT_WRIST"]]
    );
    const right_wrist_point = new Point3d(
      landmarks[landmarksDict["RIGHT_WRIST"]]
    );
    const wrist_mid_point = left_wrist_point.getMidPoint(right_wrist_point);

    const left_shoulder_point = new Point3d(
      landmarks[landmarksDict["LEFT_SHOULDER"]]
    );
    const right_shoulder_point = new Point3d(
      landmarks[landmarksDict["RIGHT_SHOULDER"]]
    );
    const shoulder_mid_point =
      left_shoulder_point.getMidPoint(right_shoulder_point);

    const left_foot_index_point = new Point3d(
      landmarks[landmarksDict["LEFT_FOOT_INDEX"]]
    );
    const right_foot_index_point = new Point3d(
      landmarks[landmarksDict["RIGHT_FOOT_INDEX"]]
    );
    const foot_index_mid_point = left_foot_index_point.getMidPoint(
      right_foot_index_point
    );

    const xDir = wrist_mid_point.subtract(foot_index_mid_point);
    const yDir = shoulder_mid_point.subtract(wrist_mid_point);
    return new CoordinateSystem3D(foot_index_mid_point, xDir, yDir);
  }

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const basis = this.getCustomBasis(landmarks);
    // const basis = CoordinateSystem3D.canonicalSystem;
    const utilLandmarks = [
      landmarksDict.LEFT_WRIST,
      landmarksDict.RIGHT_WRIST,
      landmarksDict.LEFT_ELBOW,
      landmarksDict.RIGHT_ELBOW,
      landmarksDict.LEFT_SHOULDER,
      landmarksDict.RIGHT_SHOULDER,
      landmarksDict.LEFT_HIP,
      landmarksDict.RIGHT_HIP,
      landmarksDict.LEFT_KNEE,
      landmarksDict.RIGHT_KNEE,
      landmarksDict.LEFT_ANKLE,
      landmarksDict.RIGHT_ANKLE,
    ];
    const points = utilLandmarks.map((ld) =>
      basis.toLocal(new Point3d(landmarks[ld])).toList()
    );

    const inputTensor = tensor([points]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;
    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}

export class FcnnHighPlankAnglesModel extends NeuralNetworkModel {
  modelPath = "models/high-plank/fcnn-angles/full-body-model/model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const triplets: [LandmarkKey, LandmarkKey, LandmarkKey][] = [
      ["LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER"],
      ["LEFT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "LEFT_SHOULDER"],

      ["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_HIP"],
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP"],

      ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
      ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
      ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
      ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
      ["LEFT_ANKLE", "LEFT_HIP", "RIGHT_HIP"],
      ["RIGHT_ANKLE", "RIGHT_HIP", "LEFT_HIP"],

      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "LEFT_SHOULDER"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "RIGHT_SHOULDER"],
      ["LEFT_FOOT_INDEX", "LEFT_WRIST", "RIGHT_WRIST"],
      ["RIGHT_FOOT_INDEX", "RIGHT_WRIST", "LEFT_WRIST"],
    ];
    const features = triplets.map((tp) =>
      Point3d.get_angle_from_joints_triplet(landmarks, tp)
    );
    const inputTensor = tensor([features]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;

    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}
