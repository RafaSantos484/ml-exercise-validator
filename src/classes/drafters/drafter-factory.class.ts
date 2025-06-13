import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { Constructor, Exercise } from "../../types";
import Drafter from "./drafter.class";
import HighPlankDrafter from "./high-plank-drafter.class";

export default class DrafterFactory {
  private static draftersDict: Record<Exercise, Constructor<Drafter>> = {
    high_plank: HighPlankDrafter,
  };

  private static drafters: Record<string, Drafter> = {};

  private static getDrafter(exercise: Exercise) {
    if (!this.drafters[exercise]) {
      this.drafters[exercise] = new DrafterFactory.draftersDict[exercise]();
    }
    return this.drafters[exercise];
  }

  static getDraftInfo(exercise: Exercise, landmarks: NormalizedLandmark[]) {
    const drafter = this.getDrafter(exercise);
    return drafter.getDraftInfo(landmarks);
  }
}
