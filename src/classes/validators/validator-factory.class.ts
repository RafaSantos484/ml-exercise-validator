import type { Constructor, Exercise } from "../../types";
import {
  FCNNAnglesFullBodyModel,
  FCNNPointsFullBodyModel,
} from "../models/fcnn.class";
import HighPlankValidator from "./high-plank-validator.class";
import type { Validator } from "./validator.class";

type ValidatorChild = Constructor<Validator>;

export default class ValidatorFactory {
  private static validatorsDict: Record<Exercise, ValidatorChild> = {
    high_plank: HighPlankValidator,
  };

  private static validators: Record<string, Validator> = {};

  public static getValidator(exercise: Exercise) {
    const model = new FCNNPointsFullBodyModel();
    // const model = new FCNNAnglesFullBodyModel();
    if (!this.validators[exercise]) {
      this.validators[exercise] = new ValidatorFactory.validatorsDict[exercise](
        model
      );
    }
    return this.validators[exercise];
  }
}
