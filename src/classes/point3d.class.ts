import type { Landmark } from "@mediapipe/tasks-vision";
import { landmarksDict, type LandmarkKey } from "../types";

type Point3dTuple = [number, number, number];

export default class Point3d {
  public x: number;
  public y: number;
  public z: number;

  constructor(landmark: Landmark | Point3dTuple) {
    if (Array.isArray(landmark)) {
      this.x = landmark[0];
      this.y = landmark[1];
      this.z = landmark[2];
    } else {
      this.x = landmark.x;
      this.y = landmark.y;
      this.z = landmark.z;
    }
  }

  public subtract(p2: Point3d) {
    return new Point3d([this.x - p2.x, this.y - p2.y, this.z - p2.z]);
  }

  public getMidPoint(p2: Point3d) {
    return new Point3d([
      (this.x + p2.x) / 2,
      (this.y + p2.y) / 2,
      (this.z + p2.z) / 2,
    ]);
  }

  public norm() {
    return Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2);
  }

  public normalize() {
    const norm = this.norm();
    return new Point3d([this.x / norm, this.y / norm, this.z / norm]);
  }

  public dot(p2: Point3d) {
    return this.x * p2.x + this.y * p2.y + this.z * p2.z;
  }

  public cross(p2: Point3d) {
    return new Point3d([
      this.y * p2.z - this.z * p2.y,
      this.z * p2.x - this.x * p2.z,
      this.x * p2.y - this.y * p2.x,
    ]);
  }

  public get_angle(v2: Point3d, degrees = false) {
    const dot_product = this.dot(v2);
    const norm_self = this.norm();
    const norm_other = v2.norm();
    const angle_rad = Math.acos(dot_product / (norm_self * norm_other));

    if (!degrees) {
      return angle_rad;
    }
    return angle_rad * (180 / Math.PI);
  }

  public toList() {
    return [this.x, this.y, this.z];
  }

  public toString(fractionDigits = 2) {
    return `(${this.x.toFixed(fractionDigits)}, ${this.y.toFixed(
      fractionDigits
    )}, ${this.z.toFixed(fractionDigits)})`;
  }

  public static getAngleFromPointsTriplet(
    landmarks: Landmark[],
    triplet: LandmarkKey[],
    props: { degrees?: boolean; normalize?: boolean } = {}
  ): number {
    const degrees = props.degrees ?? false;
    const normalize = props.normalize ?? true;

    const a = new Point3d(landmarks[landmarksDict[triplet[0]]]);
    const b = new Point3d(landmarks[landmarksDict[triplet[1]]]);
    const c = new Point3d(landmarks[landmarksDict[triplet[2]]]);

    const vec1 = b.subtract(a);
    const vec2 = c.subtract(b);
    let angle = vec1.get_angle(vec2, degrees);
    if (normalize) {
      if (degrees) {
        angle /= 180;
      } else {
        angle /= Math.PI;
      }
    }

    return angle;
  }
}
