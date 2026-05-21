import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "#020617",
        foreground: "#e2e8f0",
        muted: "#94a3b8",
        panel: "#0f172a",
        accent: "#38bdf8",
        success: "#22c55e",
      },
      boxShadow: {
        glow: "0 0 50px rgba(56, 189, 248, 0.18)",
      },
    },
  },
  plugins: [],
};

export default config;
