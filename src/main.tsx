import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { createTheme, CssBaseline, ThemeProvider } from "@mui/material";

export const lightTheme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1976d2",
    },
    background: {
      default: "#fff",
    },
    text: {
      primary: "#000",
    },
  },
});
export const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#90caf9",
    },
    background: {
      default: "#282727",
    },
    text: {
      primary: "#e0e0e0",
    },
  },
});

const prefersDarkMode = window.matchMedia(
  "(prefers-color-scheme: dark)"
).matches;
const theme = prefersDarkMode ? darkTheme : lightTheme;

createRoot(document.getElementById("root")!).render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <App />
  </ThemeProvider>
);
