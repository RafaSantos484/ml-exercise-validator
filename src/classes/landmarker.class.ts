import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

export class Landmarker {
  private static poseLandmarker: PoseLandmarker | null = null;

  public static async load() {
    if (!this.poseLandmarker) {
      const vision = await FilesetResolver.forVisionTasks(
        "src/assets/models/landmarker/wasm"
      );
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "src/assets/models/landmarker/pose_landmarker_full.task",
        },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    }
  }

  public static detect(videoEl: HTMLVideoElement, timestamp: number) {
    const landmarker = this.poseLandmarker;
    if (!landmarker) {
      this.load();
      return null;
    }

    return landmarker.detectForVideo(videoEl, timestamp);
  }

  public static close() {
    const landmarker = this.poseLandmarker;
    if (landmarker) {
      landmarker.close();
    }
  }
}
