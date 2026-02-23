import { FC } from "react";
import type { PersonaplexConnectionStatus } from "../hooks/usePersonaplexSession";

type ConnectButtonProps = {
  status: PersonaplexConnectionStatus;
  onConnect: () => void;
  onDisconnect: () => void;
  disabled?: boolean;
  className?: string;
};

export const ConnectButton: FC<ConnectButtonProps> = ({
  status,
  onConnect,
  onDisconnect,
  disabled = false,
  className = "",
}) => {
  const isConnected = status === "connected";

  return (
    <button
      type="button"
      onClick={isConnected ? onDisconnect : onConnect}
      disabled={disabled || status === "connecting"}
      className={`
        px-5 py-2.5 rounded-lg font-medium text-sm
        transition-all duration-200
        focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
        ${
          isConnected
            ? "bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/50"
            : "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/50"
        }
      `}
    >
      {status === "connecting"
        ? "Connecting..."
        : isConnected
          ? "Disconnect"
          : "Connect"}
    </button>
  );
};
