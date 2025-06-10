import type { Landmark } from "@mediapipe/tasks-vision";
import type { Tensor } from "@tensorflow/tfjs";

export abstract class Model {
  abstract load(): Promise<void>;
  abstract predict(landmarks: Landmark[]): Tensor | null;
}
