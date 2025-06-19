import type { Landmark } from "@mediapipe/tasks-vision";
import { KnnModel } from "../knn.class";
import Point3d from "../../point3d.class";

export class KnnHighPlankAnglesModel extends KnnModel {
  modelPath = "models/high-plank/knn/full_body_model.json";

  predict(landmarks: Landmark[]): string | null {
    if (!this.model) {
      this.load();
      return null;
    }

    // const angles = anglesExtractor.getFeatures("high_plank", landmarks);
    const angles = this.model.modelJson.features.angles.map((triplet) =>
      Point3d.getAngleFromPointsTriplet(landmarks, triplet)
    );
    return this.model.predict(angles);
  }
}
