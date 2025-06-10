import { useEffect, useRef, useState } from "react";
import { type PoseLandmarkerResult } from "@mediapipe/tasks-vision";
import {
  Button,
  CircularProgress,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";
import DrafterFactory from "./classes/drafters/drafter-factory.class";
import { Landmarker } from "./classes/landmarker.class";
import { exercisesTranslator, type Exercise } from "./types";
import { GitHub } from "@mui/icons-material";

import "./App.scss";

// import PlankImage from "./assets/plank.png";
import PushUpImage from "./assets/push-up.png";
import { ModelFactory } from "./classes/models/model.class";
// import SidePlankImage from "./assets/side-plank.png";

const exerciseImages: Record<Exercise, string> = {
  high_plank: PushUpImage,
};

function GlobalCircularProgress() {
  return (
    <div className="global-circular-progress">
      <CircularProgress size="10rem" />
    </div>
  );
}

type CameraComponentProps = {
  selectedExercise: Exercise;
  selectedModelName: string;
  close: () => void;
};

function getScreenDim() {
  return {
    width: window.innerWidth,
    height: window.innerHeight,
  };
}

function CameraComponent({
  selectedExercise,
  selectedModelName,
  close,
}: CameraComponentProps) {
  const [screenDim, setScreenDim] = useState(getScreenDim());
  const [isLoading, setIsLoading] = useState(true);
  const [exerciseValidation, setExerciseValidation] = useState<string | null>(
    null
  );

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafId = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);

  useEffect(() => {
    let isMounted = true;
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    const model = ModelFactory.getModel(selectedExercise, selectedModelName);

    const initPoseLandmarker = async () => {
      await Landmarker.load();
      await model.load();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      video.srcObject = stream;
      video.onloadeddata = () => {
        video.play();
        renderLoop();
      };
    };

    const renderLoop = () => {
      if (!isMounted) return;

      if (video.readyState >= 2) {
        const now = performance.now();
        if (video.currentTime !== lastVideoTimeRef.current) {
          const result = Landmarker.detect(video, now);
          if (result) {
            drawResults(result);
          }
          lastVideoTimeRef.current = video.currentTime;
        }
      }

      rafId.current = requestAnimationFrame(renderLoop);
    };

    const drawResults = (results: PoseLandmarkerResult) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      setIsLoading(false);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (!results.landmarks.length || !results.worldLandmarks.length) {
        setExerciseValidation(null);
        return;
      }

      const drafter = DrafterFactory.getInstance().getDrafter(selectedExercise);
      drafter.draw(results.landmarks[0], canvas, ctx);

      const validation = model.predict(results.worldLandmarks[0]);
      setExerciseValidation(validation);
    };

    initPoseLandmarker();

    return () => {
      isMounted = false;
      cancelAnimationFrame(rafId.current ?? 0);
      const stream = video.srcObject as MediaStream;
      stream?.getTracks().forEach((track) => track.stop());
      // Landmarker.close();
    };
  }, [selectedExercise, selectedModelName]);

  useEffect(() => {
    const onResize = () => setScreenDim(getScreenDim());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <>
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas
        ref={canvasRef}
        width={screenDim.width}
        height={screenDim.height}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
        }}
      />

      {isLoading ? (
        <GlobalCircularProgress />
      ) : (
        <div className="exercise-feedback-container">
          {(() => {
            let color = "";
            let message = "";
            if (exerciseValidation === null) {
              color = "yellow";
              message = "Aguardando posição";
            } else if (exerciseValidation === "") {
              color = "green";
              message = "Exercício correto!";
            } else {
              color = "red";
              message = exerciseValidation;
            }
            return <p style={{ color }}>{message}</p>;
          })()}

          <Button
            variant="contained"
            onClick={() => {
              close();
            }}
          >
            Fechar Câmera
          </Button>
        </div>
      )}
    </>
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
        <div className="select-exercise-container">
          <IconButton
            className="corner-icon"
            onClick={() => {
              window.open(
                "https://github.com/RafaSantos484/exercise-pose-validator"
              );
            }}
          >
            <GitHub />
          </IconButton>

          <h1>Validador de Exercícios</h1>

          <FormControl fullWidth sx={{ maxWidth: "350px" }}>
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
          <FormControl fullWidth sx={{ maxWidth: "350px" }}>
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

          <Button
            variant="contained"
            onClick={() => {
              if (!selectedModel) {
                alert("Selecione o modelo do exercício");
              } else {
                setIsCameraOpen(true);
              }
            }}
          >
            Abrir Câmera
          </Button>
        </div>
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
