import type { FC } from "react";
import type { ThemeMode } from "../hooks/useTheme";

type Props = {
  mode: ThemeMode;
  onToggle: () => void;
  className?: string;
};

/** ChatGPT-style: light default, optional dark via class on <html>. */
export const ThemeToggle: FC<Props> = ({ mode, onToggle, className = "" }) => {
  const isDark = mode === "dark";
  return (
    <button
      type="button"
      onClick={onToggle}
      className={
        "inline-flex items-center justify-center rounded-full border min-h-[44px] min-w-[44px] md:min-h-0 md:min-w-0 md:h-9 md:w-9 px-2 transition-colors " +
        "border-gray-100 bg-white/90 text-gray-700 shadow-sm hover:bg-white " +
        "dark:border-gray-600 dark:bg-[#343541] dark:text-gray-200 dark:hover:bg-[#404040] " +
        className
      }
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {isDark ? (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
      )}
    </button>
  );
};
