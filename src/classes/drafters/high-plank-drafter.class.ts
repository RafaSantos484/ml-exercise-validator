import type { Landmark } from "@mediapipe/tasks-vision";
import { landmarksDict } from "../../types";
import Drafter from "./drafter.class";

export default class HighPlankDrafter extends Drafter {
  constructor() {
    super([
      landmarksDict.LEFT_WRIST,
      landmarksDict.RIGHT_WRIST,
      landmarksDict.LEFT_ELBOW,
      landmarksDict.RIGHT_ELBOW,
      landmarksDict.LEFT_SHOULDER,
      landmarksDict.RIGHT_SHOULDER,
      landmarksDict.LEFT_HIP,
      landmarksDict.RIGHT_HIP,
      landmarksDict.LEFT_KNEE,
      landmarksDict.RIGHT_KNEE,
      landmarksDict.LEFT_ANKLE,
      landmarksDict.RIGHT_ANKLE,
    ]);
  }

  public draw(
    landmarks: Landmark[],
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D
  ) {
    super.draw(landmarks, canvas, ctx);
    const points = this.getPointsFromSelectedLandmarks(landmarks);

    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_WRIST],
      points[landmarksDict.LEFT_ELBOW]
    );
    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.RIGHT_WRIST],
      points[landmarksDict.RIGHT_ELBOW]
    );

    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_ELBOW],
      points[landmarksDict.LEFT_SHOULDER]
    );
    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.RIGHT_ELBOW],
      points[landmarksDict.RIGHT_SHOULDER]
    );

    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_SHOULDER],
      points[landmarksDict.LEFT_HIP]
    );
    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.RIGHT_SHOULDER],
      points[landmarksDict.RIGHT_HIP]
    );

    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_HIP],
      points[landmarksDict.LEFT_KNEE]
    );
    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.RIGHT_HIP],
      points[landmarksDict.RIGHT_KNEE]
    );

    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_KNEE],
      points[landmarksDict.LEFT_ANKLE]
    );
    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.RIGHT_KNEE],
      points[landmarksDict.RIGHT_ANKLE]
    );

    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_SHOULDER],
      points[landmarksDict.RIGHT_SHOULDER]
    );
    this.drawLine(
      canvas,
      ctx,
      points[landmarksDict.LEFT_HIP],
      points[landmarksDict.RIGHT_HIP]
    );
  }
}
