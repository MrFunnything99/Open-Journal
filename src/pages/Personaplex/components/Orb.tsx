import { FC, useMemo } from "react";

export type OrbState = "idle" | "userSpeaking" | "aiThinking" | "aiSpeaking";

type OrbProps = {
  state: OrbState;
  /** 0–1 progress through 4 task quarters when state is "aiThinking" */
  thinkingProgress?: number;
  className?: string;
};

const ORB_R = 47;
const ORB_CIRCUMFERENCE = 2 * Math.PI * ORB_R;
const ORB_QUARTER = ORB_CIRCUMFERENCE / 4;

export const Orb: FC<OrbProps> = ({ state, thinkingProgress = 0, className = "" }) => {
  const getPulseClass = () => {
    switch (state) {
      case "userSpeaking":
        return "animate-orb-user";
      case "aiSpeaking":
        return "animate-orb-ai";
      case "aiThinking":
        return "animate-orb-idle";
      default:
        return "animate-orb-idle";
    }
  };

  const getGlowClass = () => {
    switch (state) {
      case "userSpeaking":
        return "shadow-[0_0_60px_rgba(34,197,94,0.4)]";
      case "aiSpeaking":
        return "shadow-[0_0_60px_rgba(16,163,127,0.38)]";
      case "aiThinking":
        return "shadow-[0_0_44px_rgba(16,185,129,0.22)] dark:shadow-[0_0_40px_rgba(107,114,128,0.2)]";
      default:
        /** Light mode: soft mint halo (variant 1); dark: neutral */
        return "shadow-[0_0_50px_rgba(16,185,129,0.3)] dark:shadow-[0_0_40px_rgba(107,114,128,0.12)]";
    }
  };

  const getInnerGlowClass = () => {
    switch (state) {
      case "userSpeaking":
        return "from-emerald-500/30 to-cyan-500/20";
      case "aiSpeaking":
        return "from-[#10a37f]/30 to-emerald-600/20";
      case "aiThinking":
        return "from-gray-400/20 to-gray-500/10 dark:from-slate-400/15 dark:to-slate-600/10";
      default:
        return "from-emerald-100/40 to-gray-200/20 dark:from-slate-400/10 dark:to-slate-600/5";
    }
  };

  const thinkingFilledQuarters = useMemo(() => {
    if (state !== "aiThinking" || thinkingProgress <= 0) return 0;
    return Math.min(4, Math.ceil(thinkingProgress * 4));
  }, [state, thinkingProgress]);

  const thinkingStrokeDash = useMemo(() => {
    const filled = thinkingFilledQuarters;
    const drawn = filled * ORB_QUARTER;
    const gap = (4 - filled) * ORB_QUARTER;
    return `${drawn} ${gap}`;
  }, [thinkingFilledQuarters]);

  return (
    <div
      className={`relative flex items-center justify-center ${className}`}
      aria-hidden
    >
      <div
        className={`
          absolute rounded-full bg-gradient-to-br ${getInnerGlowClass()}
          w-24 h-24 sm:w-32 sm:h-32 md:w-80 md:h-80
          ${getPulseClass()} ${getGlowClass()}
          transition-all duration-700 ease-in-out
        `}
      />
      <div
        className={`
          relative rounded-full
          w-20 h-20 sm:w-28 sm:h-28 md:w-64 md:h-64
          bg-white shadow-inner dark:bg-gradient-to-br dark:from-slate-800/90 dark:to-slate-900/95
          border border-gray-100 dark:border-slate-700/50
          backdrop-blur-sm
          ${getPulseClass()}
        `}
      />
      {state === "aiThinking" && (
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 sm:w-28 sm:h-28 md:w-64 md:h-64 pointer-events-none"
          aria-hidden
        >
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r={ORB_R}
              fill="none"
              stroke="currentColor"
              strokeWidth="4"
              strokeDasharray={thinkingStrokeDash}
              strokeLinecap="butt"
              className="text-gray-800 dark:text-white/90 transition-all duration-300 ease-out"
            />
          </svg>
        </div>
      )}
      <div
        className={`
          absolute inset-0 rounded-full
          bg-gradient-to-br from-white/5 to-transparent
          pointer-events-none
        `}
      />
    </div>
  );
};
