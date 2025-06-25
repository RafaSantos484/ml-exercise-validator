import { Model } from "./model.class";
import type { Landmark } from "@mediapipe/tasks-vision";
import { type InferenceSession } from "onnxruntime-web";
import Utils from "../utils.class";

export class LogisticRegressionModel extends Model {
  protected modelPath =
    "/models/high-plank/logistic-regression/full_body_model.onnx";

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
