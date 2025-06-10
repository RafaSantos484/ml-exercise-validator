import type { Constructor, Exercise } from "../../types";
import Drafter from "./drafter.class";
import HighPlankDrafter from "./high-plank-drafter.class";

export default class DrafterFactory {
  private static draftersDict: Record<Exercise, Constructor<Drafter>> = {
    high_plank: HighPlankDrafter,
  };
  private static instance: DrafterFactory | undefined = undefined;

  private drafters: Record<string, Drafter>;

  private constructor() {
    this.drafters = {};
  }

  public static getInstance() {
    if (!DrafterFactory.instance) {
      DrafterFactory.instance = new DrafterFactory();
    }
    return DrafterFactory.instance;
  }

  public getDrafter(exercise: Exercise) {
    if (!this.drafters[exercise]) {
      this.drafters[exercise] = new DrafterFactory.draftersDict[exercise]();
    }
    return this.drafters[exercise];
  }
}
