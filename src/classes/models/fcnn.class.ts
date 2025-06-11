import {
  loadLayersModel,
  regularizers,
  serialization,
  tensor,
  Tensor,
  type LayersModel,
} from "@tensorflow/tfjs";
import type { Model } from "./model.class";
import { landmarksDict, type LandmarkKey } from "../../types";
import Point3d from "../point3d.class";
import type { Landmark } from "@mediapipe/tasks-vision";
import type { L1L2Args } from "@tensorflow/tfjs-layers/dist/regularizers";
import CoordinateSystem3D from "../coordinate-system-3d.class";

class L2 {
  static className = "L2";

  constructor(config: L1L2Args) {
    return regularizers.l1l2(config);
  }
}
serialization.registerClass(L2 as any);

abstract class FcnnModel implements Model {
  abstract modelPath: string;
  protected model: LayersModel | null = null;

  public async load() {
    if (!this.model) {
      this.model = await loadLayersModel(this.modelPath);
    }
  }
  abstract predict(landmarks: Landmark[]): string | null;
}

export class FcnnHighPlankPointsFullBodyModel extends FcnnModel {
  modelPath =
    "src/assets/models/fcnn-high-plank-points/full-body-model/model.json";

  private static getCustomBasis(landmarks: Landmark[]) {
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

    const basis = FcnnHighPlankPointsFullBodyModel.getCustomBasis(landmarks);
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

    const leftShoulderPoint = new Point3d(
      landmarks[landmarksDict.LEFT_SHOULDER]
    );
    const rightShoulderPoint = new Point3d(
      landmarks[landmarksDict.RIGHT_SHOULDER]
    );
    const shoulderMidPoint = leftShoulderPoint.getMidPoint(rightShoulderPoint);
    const leftAnklePoint = new Point3d(landmarks[landmarksDict.LEFT_ANKLE]);
    const rightAnklePoint = new Point3d(landmarks[landmarksDict.RIGHT_ANKLE]);
    const ankleMidPoint = leftAnklePoint.getMidPoint(rightAnklePoint);
    const height = shoulderMidPoint.subtract(ankleMidPoint).norm();
    const normalizedPoints = points.map((p) => [
      p[0] / height,
      p[1] / height,
      p[2] / height,
    ]);

    const inputTensor = tensor([normalizedPoints]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;

    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}
export class CnnHighPlankPointsFullBodyModel extends FcnnHighPlankPointsFullBodyModel {
  modelPath =
    "src/assets/models/cnn-high-plank-points/full-body-model/model.json";
}

export class FcnnHighPlankAnglesFullBodyModel extends FcnnModel {
  modelPath =
    "src/assets/models/fcnn-high-plank-angles/full-body-model/model.json";

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
    const inputTensor = tensor([features]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;

    const predictionArray = outputTensor.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];
    return `${predictedClass}(${maxProb.toFixed(2)})`;
  }
}
