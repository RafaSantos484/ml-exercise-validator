import type { Landmark } from "@mediapipe/tasks-vision";
import { RandomForestModel } from "../random-forest.class";
import Point3d from "../../point3d.class";

export class RandomForestHighPlankAnglesModel extends RandomForestModel {
  modelPath = "models/high-plank/random-forest/full_body_model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    const angles = this.model.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    return this.model.predict(angles);
  }
}
