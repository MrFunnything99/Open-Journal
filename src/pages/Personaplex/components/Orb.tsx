import { FC } from "react";

export type OrbState = "idle" | "userSpeaking" | "aiSpeaking";

type OrbProps = {
  state: OrbState;
  className?: string;
};

export const Orb: FC<OrbProps> = ({ state, className = "" }) => {
  const getPulseClass = () => {
    switch (state) {
      case "userSpeaking":
        return "animate-orb-user";
      case "aiSpeaking":
        return "animate-orb-ai";
      default:
        return "animate-orb-idle";
    }
  };

  const getGlowClass = () => {
    switch (state) {
      case "userSpeaking":
        return "shadow-[0_0_60px_rgba(34,197,94,0.4)]";
      case "aiSpeaking":
        return "shadow-[0_0_60px_rgba(168,85,247,0.4)]";
      default:
        return "shadow-[0_0_40px_rgba(148,163,184,0.15)]";
    }
  };

  const getInnerGlowClass = () => {
    switch (state) {
      case "userSpeaking":
        return "from-emerald-500/30 to-cyan-500/20";
      case "aiSpeaking":
        return "from-violet-500/30 to-fuchsia-500/20";
      default:
        return "from-slate-400/10 to-slate-600/5";
    }
  };

  return (
    <div
      className={`relative flex items-center justify-center ${className}`}
      aria-hidden
    >
      <div
        className={`
          absolute rounded-full bg-gradient-to-br ${getInnerGlowClass()}
          w-32 h-32 md:w-80 md:h-80
          ${getPulseClass()} ${getGlowClass()}
          transition-all duration-700 ease-in-out
        `}
      />
      <div
        className={`
          relative rounded-full
          w-28 h-28 md:w-64 md:h-64
          bg-gradient-to-br from-slate-800/90 to-slate-900/95
          border border-slate-700/50
          backdrop-blur-sm
          ${getPulseClass()}
        `}
      />
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
