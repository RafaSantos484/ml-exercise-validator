import {
  FilesetResolver,
  PoseLandmarker,
  type ImageSource,
} from "@mediapipe/tasks-vision";

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

  private static getImageDataFromVideoSync(
    videoElement: HTMLVideoElement
  ): ImageData {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      throw new Error("Não foi possível obter o contexto 2D do canvas");
    }

    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
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
