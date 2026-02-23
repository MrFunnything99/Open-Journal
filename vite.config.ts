import { defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  const apiTarget = env.VITE_API_URL || "http://localhost:3001";
  return {
    server: {
      host: "0.0.0.0",
      https: false,
      proxy: {
        "/api": { target: apiTarget, changeOrigin: true },
      },
    },
    plugins: [
      topLevelAwait({
        // The export name of top-level await promise for each chunk module
        promiseExportName: "__tla",
        // The function to generate import names of top-level await promise in each chunk module
        promiseImportName: i => `__tla_${i}`,
      }),
    ],
  };
});
