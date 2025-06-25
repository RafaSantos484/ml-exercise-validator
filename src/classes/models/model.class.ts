import type { Landmark } from "@mediapipe/tasks-vision";
import type { LandmarkKey } from "../../types";
import Point3d from "../point3d.class";
import { InferenceSession, Tensor } from "onnxruntime-web";
import Utils from "../utils.class";

export interface Classifier {
  predict(landmarks: Landmark[]): Promise<string>;
  load(): Promise<void>;
}

export class SklearnModel implements Classifier {
  protected modelPath: string;
  private session: InferenceSession | undefined;

  constructor(modelPath: string) {
    this.modelPath = modelPath;
  }

  async load(): Promise<void> {
    if (!this.session) {
      this.session = await InferenceSession.create(this.modelPath);
    }
  }

  protected async getSession() {
    await this.load();
    return this.session as InferenceSession;
  }

  getTensor(landmarks: Landmark[]) {
    const triplets: LandmarkKey[][] = [
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
    const angles = triplets.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    const data = Float32Array.from(angles);
    const dims = [1, angles.length];
    const tensor = new Tensor("float32", data, dims);
    return tensor;
  }

  async predict(landmarks: Landmark[]): Promise<string> {
    const tensor = this.getTensor(landmarks);
    const session = await this.getSession();
    const feeds: InferenceSession.FeedsType = {
      [session.inputNames[0]]: tensor,
    };
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    const label = output.data[0] as string;
    return Utils.translate(label);
  }
}

export class KerasModel extends SklearnModel {
  protected classes: string[];

  constructor(modelPath: string, classes: string[]) {
    super(modelPath);
    this.classes = classes;
  }

  async predict(landmarks: Landmark[]): Promise<string> {
    const tensor = this.getTensor(landmarks);
    const session = await this.getSession();
    const feeds: InferenceSession.FeedsType = {
      [session.inputNames[0]]: tensor,
    };
    const results = await session.run(feeds);
    const probs = results[session.outputNames[0]].data as Float32Array;
    const maxIdx = probs.indexOf(Math.max(...probs));
    const label = this.classes[maxIdx];
    return Utils.translate(label);
  }
}
