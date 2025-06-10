import { Validator } from "./validator.class";
import type { Landmark } from "@mediapipe/tasks-vision";

export default class HighPlankValidator extends Validator {
  public validate(landmarks: Landmark[]): [string, number] | null {
    const prediction = this.model.predict(landmarks);
    if (!prediction) {
      return null;
    }

    const predictionArray = prediction.dataSync();
    const maxProb = Math.max(...predictionArray);
    const predictedIndex = predictionArray.indexOf(maxProb);
    const predictedClass = ["Incorreto", "Correto"][predictedIndex];

    return [predictedClass, maxProb];
  }
}
