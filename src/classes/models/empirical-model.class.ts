import type { Landmark } from "@mediapipe/tasks-vision";
import type { Classifier, ValidationResult } from "../../types";
import Point3d from "../point3d.class";
import Utils from "../utils.class";

export class EmpiricalModel implements Classifier {
  private static readonly anglesIntervals = {
    wrist_elbow_shoulder: [0, 45],
    wrist_shoulders: [75, 100],
    wrist_shoulder_hip: [80, 130],
    shoulder_hip_knee: [0, 25],
    hip_knee_ankle: [0, 25],
  };

  async load() {}

  async predict(landmarks: Landmark[]): Promise<ValidationResult> {
    let [min, max] = [0, 0];

    const leftWristElbowShoulderAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER"],
      { degrees: true, normalize: false }
    );
    const rightWristElbowShoulderAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER"],
      { degrees: true, normalize: false }
    );
    [min, max] = EmpiricalModel.anglesIntervals.wrist_elbow_shoulder;
    if (
      !Utils.isBetween(leftWristElbowShoulderAngle, min, max) ||
      !Utils.isBetween(rightWristElbowShoulderAngle, min, max)
    ) {
      return {
        text: "Alinhe os cotovelos com os punhos e ombros",
        color: "red",
        isCorrect: false,
      };
    }

    const leftWristShouldersAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["LEFT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
      { degrees: true, normalize: false }
    );
    const rightWristShouldersAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "LEFT_SHOULDER"],
      { degrees: true, normalize: false }
    );
    [min, max] = EmpiricalModel.anglesIntervals.wrist_shoulders;
    if (
      !Utils.isBetween(leftWristShouldersAngle, min, max) ||
      !Utils.isBetween(rightWristShouldersAngle, min, max)
    ) {
      return {
        text: "Os braços estão muito abertos ou fechados",
        color: "red",
        isCorrect: false,
      };
    }

    const leftWristShoulderHipAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_HIP"],
      { degrees: true, normalize: false }
    );
    const rightWristShoulderHipAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP"],
      { degrees: true, normalize: false }
    );
    [min, max] = EmpiricalModel.anglesIntervals.wrist_shoulder_hip;
    if (
      !Utils.isBetween(leftWristShoulderHipAngle, min, max) ||
      !Utils.isBetween(rightWristShoulderHipAngle, min, max)
    ) {
      return {
        text: "Mantenha os punhos alinhados com os ombros e quadris",
        color: "red",
        isCorrect: false,
      };
    }

    const leftShoulderHipKneeAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
      { degrees: true, normalize: false }
    );
    const rightShoulderHipKneeAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
      { degrees: true, normalize: false }
    );
    [min, max] = EmpiricalModel.anglesIntervals.shoulder_hip_knee;
    if (
      !Utils.isBetween(leftShoulderHipKneeAngle, min, max) ||
      !Utils.isBetween(rightShoulderHipKneeAngle, min, max)
    ) {
      return {
        text: "Mantenha os joelhos alinhados com os quadris e ombros",
        color: "red",
        isCorrect: false,
      };
    }

    const leftHipKneeAnkleAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
      { degrees: true, normalize: false }
    );
    const rightHipKneeAnkleAngle = Point3d.getAngleFromPointsTriplet(
      landmarks,
      ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
      { degrees: true, normalize: false }
    );
    [min, max] = EmpiricalModel.anglesIntervals.hip_knee_ankle;
    if (
      !Utils.isBetween(leftHipKneeAnkleAngle, min, max) ||
      !Utils.isBetween(rightHipKneeAnkleAngle, min, max)
    ) {
      return {
        text: "Mantenha os tornozelos alinhados com os quadris e joelhos",
        color: "red",
        isCorrect: false,
      };
    }

    const result: ValidationResult = {
      text: "Correto! Continue assim!",
      color: "green",
      isCorrect: true,
    };
    return result;
  }
}
