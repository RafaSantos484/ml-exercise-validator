import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

export class Landmarker {
  private static poseLandmarker: PoseLandmarker | undefined;

  public static async load() {
    if (!this.poseLandmarker) {
      const vision = await FilesetResolver.forVisionTasks(
        "models/landmarker/wasm"
      );
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "models/landmarker/pose_landmarker_full.task",
          // modelAssetPath: "models/landmarker/pose_landmarker_lite.task",
        },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    }
  }

  public static async getLandmarker() {
    await this.load();
    return this.poseLandmarker as PoseLandmarker;
  }

  public static async detect(videoEl: HTMLVideoElement, timestamp: number) {
    const landmarker = await this.getLandmarker();
    return landmarker.detectForVideo(videoEl, timestamp);
  }

  public static close() {
    if (this.poseLandmarker) {
      this.poseLandmarker.close();
    }
    this.poseLandmarker = undefined;
  }
}
