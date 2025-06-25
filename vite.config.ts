import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

// https://vite.dev/config/
export default defineConfig({
  server: {
    host: "0.0.0.0", // permite acesso externo
    port: 5173,
    allowedHosts: true,
  },
  plugins: [react(), basicSsl()],
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
