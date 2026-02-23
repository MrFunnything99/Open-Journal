import { FC } from "react";
import type { PersonaplexConnectionStatus } from "../hooks/usePersonaplexSession";

type ConnectionStatusProps = {
  status: PersonaplexConnectionStatus;
  className?: string;
};

export const ConnectionStatus: FC<ConnectionStatusProps> = ({
  status,
  className = "",
}) => {
  const config = {
    connected: {
      color: "bg-emerald-500",
      pulse: "animate-pulse",
      label: "Online",
    },
    disconnected: {
      color: "bg-red-500",
      pulse: "",
      label: "Offline",
    },
    connecting: {
      color: "bg-amber-500",
      pulse: "animate-pulse",
      label: "Connecting...",
    },
    error: {
      color: "bg-red-500",
      pulse: "animate-pulse",
      label: "Error",
    },
  };

  const { color, pulse, label } = config[status];

  return (
    <div
      className={`flex items-center gap-2 text-sm text-slate-400 ${className}`}
      role="status"
      aria-live="polite"
    >
      <span
        className={`inline-block w-2 h-2 rounded-full ${color} ${pulse}`}
        aria-hidden
      />
      <span>{label}</span>
    </div>
  );
};
