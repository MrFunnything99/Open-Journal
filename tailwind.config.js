/** @type {import('tailwindcss').Config} */

export default {
  darkMode: "class",
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./index.html"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
      keyframes: {
        heroModeDesc: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        composerSlideUp: {
          "0%": { opacity: "0", transform: "translateY(24px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        chatAreaFadeIn: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "hero-mode-desc": "heroModeDesc 0.35s ease-out forwards",
        "composer-enter": "composerSlideUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.08s both",
        "chat-fade-in": "chatAreaFadeIn 0.45s cubic-bezier(0.16,1,0.3,1) both",
      },
    },
  },
  plugins: [require('daisyui')],
};
