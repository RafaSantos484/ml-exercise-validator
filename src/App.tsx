import { useEffect, useMemo, useRef, useState } from "react";
import {
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import {
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";
import DrafterFactory from "./classes/drafters/drafter-factory.class";
import { Landmarker } from "./classes/models/landmarker.class";
import { exercisesTranslator, type Exercise } from "./types";
import { GitHub } from "@mui/icons-material";

import "./App.scss";

import PushUpImage from "./assets/push-up.png";
import Webcam from "react-webcam";
import { ModelFactory } from "./classes/models/model-factory.class";
import Utils from "./classes/utils.class";
import type { ValidationResult } from "./classes/models/model.class";

const exerciseImages: Record<Exercise, string> = {
  high_plank: PushUpImage,
};

function GlobalCircularProgress() {
  return (
    <div className="fullscreen-container">
      <CircularProgress size="10rem" />
    </div>
  );
}

type CameraComponentProps = {
  selectedExercise: Exercise;
  selectedModelName: string;
  close: () => void;
};

const defaultValidationResult: ValidationResult = {
  text: "Aguardando posição",
  color: "yellow",
};
function CameraComponent({
  selectedExercise,
  selectedModelName,
  close,
}: CameraComponentProps) {
  const [screenDim, setScreenDim] = useState({ width: 0, height: 0 });

  const [isLoadingVideo, setIsLoadingVideo] = useState(true);
  const [isLoadingLandmarker, setIsLoadingLandmarker] = useState(true);
  const [isLoadingModel, setIsLoadingModel] = useState(true);

  const [exerciseValidation, setExerciseValidation] = useState(
    defaultValidationResult
  );
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[]>([]);
  const [connections, setConnections] = useState<
    [NormalizedLandmark, NormalizedLandmark][]
  >([]);

  const webcamRef = useRef<Webcam>(null);
  const rafIdRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);

  // const [fps, setFps] = useState<number | null>(null);
  const prevFpsTimestampRef = useRef<number | null>(null);
  // const [avgDetectionTime, setAvgDetectionTime] = useState<number | null>(null);
  // const [avgPredictionTime, setAvgPredictionTime] = useState<number | null>(null);

  const times = useRef<Record<string, number[]>>({
    fps: [],
    detection: [],
    prediction: [],
  });

  const model = useMemo(() => {
    setIsLoadingModel(true);
    const model = ModelFactory.getModel(selectedExercise, selectedModelName);
    model.load().then(() => {
      setIsLoadingModel(false);
    });
    return model;
  }, [selectedExercise, selectedModelName]);
  const drafter = useMemo(() => {
    return DrafterFactory.getDrafter(selectedExercise);
  }, [selectedExercise]);

  const loadedEverything = useMemo(() => {
    return !isLoadingVideo && !isLoadingLandmarker && !isLoadingModel;
  }, [isLoadingVideo, isLoadingLandmarker, isLoadingModel]);

  useEffect(() => {
    Landmarker.load().then(() => {
      setIsLoadingLandmarker(false);
    });

    const loopFunc = function () {
      if (webcamRef.current?.stream?.active) {
        setIsLoadingVideo(false);
      } else {
        setTimeout(function () {
          loopFunc();
        }, 100);
      }
    };
    loopFunc();
  }, []);

  useEffect(() => {
    if (!loadedEverything) return;

    let isMounted = true;
    const rafId = rafIdRef.current;
    const video = webcamRef.current!.video!;
    const stream = webcamRef.current!.stream!;

    const initPoseLandmarker = async function () {
      await model.load();

      video.onloadeddata = function () {
        video.play().then(() => {
          renderLoop();
        });
      };
      video.srcObject = stream;
      // video.ontimeupdate = renderLoop;
    };

    const renderLoop = () => {
      if (!isMounted) return;

      if (
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        const now = performance.now();
        if (prevFpsTimestampRef.current !== null) {
          const elapsed = now - prevFpsTimestampRef.current;
          const _fps = 1000 / elapsed;
          // setFps(Math.round(_fps));
          times.current.fps.push(_fps);
          if (times.current.fps.length > 100) {
            times.current.fps.shift();
          }
        }
        prevFpsTimestampRef.current = now;

        const detectionStart = performance.now();
        const result = Landmarker.detect(video, now);
        const detectionEnd = performance.now();
        // setAvgDetectionTime(detectionEnd - detectionStart);
        times.current.detection.push(detectionEnd - detectionStart);
        if (times.current.detection.length > 100) {
          times.current.detection.shift();
        }

        if (result && result.landmarks.length && result.worldLandmarks.length) {
          setScreenDim({
            width: video.clientWidth,
            height: video.clientHeight,
          });

          drawResults(result)
            .then(() => {
              if (isMounted) {
                lastVideoTimeRef.current = video.currentTime;
              }
            })
            .finally(() => {
              if (isMounted) {
                rafIdRef.current = requestAnimationFrame(renderLoop);
              }
            });

          return;
        } else {
          setLandmarks([]);
          setConnections([]);
          setExerciseValidation(defaultValidationResult);
          // setFps(null);
          prevFpsTimestampRef.current = null;
          // setAvgDetectionTime(null);
          // setAvgPredictionTime(null);
          times.current = {
            fps: [],
            detection: [],
            prediction: [],
          };
        }
      }

      rafIdRef.current = requestAnimationFrame(renderLoop);
    };

    async function drawResults(results: PoseLandmarkerResult) {
      const [utilLandmarks, conenctions] = drafter.getDraftInfo(
        results.landmarks[0]
      );
      setLandmarks(utilLandmarks);
      setConnections(conenctions);

      const predictionStart = performance.now();
      const validation = await model.predict(results.worldLandmarks[0]);
      const predictionEnd = performance.now();
      // setAvgPredictionTime(predictionEnd - predictionStart);
      times.current.prediction.push(predictionEnd - predictionStart);
      if (times.current.prediction.length > 100) {
        times.current.prediction.shift();
      }

      setExerciseValidation(validation);
    }

    initPoseLandmarker();

    const { prediction, detection, fps } = times.current;
    return () => {
      isMounted = false;
      cancelAnimationFrame(rafId ?? 0);
      const stream = video.srcObject as MediaStream;
      stream?.getTracks().forEach((track) => track.stop());
      // Landmarker.close();

      const fpsMeanStdDev = Utils.getMeanAndStdDev(fps);
      const fpsStr = `FPS: ${fpsMeanStdDev.mean.toFixed(
        2
      )} ± ${fpsMeanStdDev.stdDev.toFixed(2)}`;
      const detectionMeanStdDev = Utils.getMeanAndStdDev(detection);
      const detectionStr = `Detecção: ${detectionMeanStdDev.mean.toFixed(
        2
      )}ms ± ${detectionMeanStdDev.stdDev.toFixed(2)}ms`;
      const predictionMeanStdDev = Utils.getMeanAndStdDev(prediction);
      const predictionStr = `Predição: ${predictionMeanStdDev.mean.toFixed(
        2
      )}ms ± ${predictionMeanStdDev.stdDev.toFixed(2)}ms`;
      alert(fpsStr + "\n" + detectionStr + "\n" + predictionStr);
    };
  }, [model, drafter, loadedEverything]);

  return (
    <div className="camera-container">
      <Webcam
        ref={webcamRef}
        style={{ display: loadedEverything ? undefined : "none" }}
        audio={false}
        videoConstraints={{ facingMode: { ideal: "environment" } }}
      />

      {!loadedEverything ? (
        <GlobalCircularProgress />
      ) : (
        <>
          {landmarks.map((landmark, idx) => (
            <div
              key={`point-${idx}`}
              className="landmark-dot"
              style={{
                top: `${landmark.y * screenDim.height}px`,
                left: `${landmark.x * screenDim.width}px`,
              }}
            ></div>
          ))}
          {connections.map(([ld1, ld2], idx) => {
            const x1 = ld1.x * screenDim.width;
            const y1 = ld1.y * screenDim.height;
            const x2 = ld2.x * screenDim.width;
            const y2 = ld2.y * screenDim.height;

            const dx = x2 - x1;
            const dy = y2 - y1;
            const length = Math.sqrt(dx ** 2 + dy ** 2);
            const angle = Math.atan2(dy, dx);

            return (
              <div
                key={`line-${idx}`}
                className="landmark-line"
                style={{
                  width: `${length}px`,
                  transform: `rotate(${angle}rad)`,
                  top: `${y1}px`,
                  left: `${x1}px`,
                }}
              />
            );
          })}

          <div className="exercise-feedback-container">
            <div className="text-container">
              {/*<span style={{ position: "absolute", top: 0, left: "1rem" }}>
                FPS: {fps}
                <br />
                Landmarking: {avgDetectionTime?.toFixed(2)}ms
                <br />
                Predição: {avgPredictionTime?.toFixed(2)}ms
              </span> */}

              <p style={{ color: exerciseValidation.color, margin: "auto" }}>
                {exerciseValidation.text}
              </p>
            </div>

            <Button
              variant="contained"
              onClick={() => {
                close();
              }}
            >
              Fechar Câmera
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

export default function App() {
  const [isCameraOpen, setIsCameraOpen] = useState(false);

  const [selectedExercise, setSelecteExercise] =
    useState<Exercise>("high_plank");
  const [selectedModel, setSelectedModel] = useState("");

  const isCameraOpenRef = useRef(isCameraOpen);

  useEffect(() => {
    isCameraOpenRef.current = isCameraOpen;
  }, [selectedExercise, isCameraOpen]);

  useEffect(() => {
    const onResize = () => {
      const vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty("--vh", `${vh}px`);
    };
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    setSelectedModel("");
  }, [selectedExercise]);

  return (
    <div className="app-container">
      {!isCameraOpen && (
        <>
          <a
            className="corner-icon"
            href="https://github.com/RafaSantos484/ml-exercise-validator"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GitHub />
          </a>

          <h1>Validador de Exercícios</h1>

          <form
            className="select-exercise-form"
            onSubmit={(e) => {
              e.preventDefault();
              setIsCameraOpen(true);
            }}
          >
            <FormControl fullWidth required>
              <InputLabel id="select-exercise-label">Exercício</InputLabel>
              <Select
                labelId="select-exercise-label"
                value={selectedExercise}
                label="Exercício"
                onChange={(e) => {
                  setSelecteExercise(e.target.value as Exercise);
                }}
              >
                {Object.entries(exercisesTranslator).map(
                  ([excercise, translatedExercise]) => (
                    <MenuItem key={excercise} value={excercise}>
                      <div className="select-menu-item">
                        <span>{translatedExercise}</span>
                        <img
                          src={exerciseImages[excercise as Exercise]}
                          alt={translatedExercise}
                        />
                      </div>
                    </MenuItem>
                  )
                )}
              </Select>
            </FormControl>
            <FormControl fullWidth required>
              <InputLabel id="select-modelname-label">Modelo</InputLabel>
              <Select
                labelId="select-modelname-label"
                value={selectedModel}
                label="Exercício"
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                }}
              >
                {ModelFactory.getExerciseModelNames(selectedExercise).map(
                  (modelName) => (
                    <MenuItem key={modelName} value={modelName}>
                      <div className="select-menu-item">
                        <span>{modelName}</span>
                        {/*<img
                        src={exerciseImages[excercise as Exercise]}
                        alt={translatedExercise}
                      /> */}
                      </div>
                    </MenuItem>
                  )
                )}
              </Select>
            </FormControl>

            <Button variant="contained" type="submit">
              Abrir Câmera
            </Button>
          </form>
        </>
      )}
      {isCameraOpen && (
        <CameraComponent
          selectedExercise={selectedExercise}
          selectedModelName={selectedModel}
          close={() => setIsCameraOpen(false)}
        />
      )}
    </div>
  );
}
