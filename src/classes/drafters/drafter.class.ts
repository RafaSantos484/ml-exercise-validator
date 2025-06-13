import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export default class Drafter {
  protected selectedLandmarksIdxs: number[];
  protected connections: [number, number][];

  constructor(selectedLandmarks?: number[], connections?: [number, number][]) {
    this.selectedLandmarksIdxs = selectedLandmarks ?? [];
    this.connections = connections ?? [];
  }

  public getDraftInfo(
    landmarks: NormalizedLandmark[]
  ): [NormalizedLandmark[], [NormalizedLandmark, NormalizedLandmark][]] {
    const selectedLandmarks = this.selectedLandmarksIdxs.map(
      (idx) => landmarks[idx]
    );
    const connections: [NormalizedLandmark, NormalizedLandmark][] =
      this.connections.map(([idx1, idx2]) => [
        landmarks[idx1],
        landmarks[idx2],
      ]);

    return [selectedLandmarks, connections];
  }
}
