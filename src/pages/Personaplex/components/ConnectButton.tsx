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
        px-6 py-2.5 rounded-full font-medium text-sm
        transition-all duration-200
        focus:outline-none focus:ring-2 focus:ring-emerald-400/60 focus:ring-offset-2 focus:ring-offset-transparent
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
        ${
          isConnected
            ? "bg-red-500/15 text-red-600 hover:bg-red-500/25 border border-red-400/40 dark:text-red-400 dark:border-red-500/50"
            : "bg-emerald-500 text-white shadow-sm hover:bg-emerald-600 border border-emerald-600/30"
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
