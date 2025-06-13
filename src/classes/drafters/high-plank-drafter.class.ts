import { landmarksDict } from "../../types";
import Drafter from "./drafter.class";

export default class HighPlankDrafter extends Drafter {
  constructor() {
    const selectedLds = [
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
    ];
    const connections: [number, number][] = [
      [landmarksDict.LEFT_WRIST, landmarksDict.LEFT_ELBOW],
      [landmarksDict.RIGHT_WRIST, landmarksDict.RIGHT_ELBOW],
      [landmarksDict.LEFT_ELBOW, landmarksDict.LEFT_SHOULDER],
      [landmarksDict.RIGHT_ELBOW, landmarksDict.RIGHT_SHOULDER],
      [landmarksDict.LEFT_SHOULDER, landmarksDict.RIGHT_SHOULDER],
      [landmarksDict.LEFT_SHOULDER, landmarksDict.LEFT_HIP],
      [landmarksDict.RIGHT_SHOULDER, landmarksDict.RIGHT_HIP],
      [landmarksDict.LEFT_HIP, landmarksDict.RIGHT_HIP],
      [landmarksDict.LEFT_HIP, landmarksDict.LEFT_KNEE],
      [landmarksDict.RIGHT_HIP, landmarksDict.RIGHT_KNEE],
      [landmarksDict.LEFT_KNEE, landmarksDict.LEFT_ANKLE],
      [landmarksDict.RIGHT_KNEE, landmarksDict.RIGHT_ANKLE],
    ];

    super(selectedLds, connections);
  }
}
