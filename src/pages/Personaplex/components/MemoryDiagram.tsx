import { useCallback, useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

type MemoryDiagramProps = {
  onRefreshStats?: () => void;
};

let mermaidInitialized = false;

function initMermaid() {
  if (mermaidInitialized) return;
  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    themeVariables: {
      primaryColor: "#7c3aed",
      primaryTextColor: "#e2e8f0",
      primaryBorderColor: "#6366f1",
      lineColor: "#94a3b8",
      secondaryColor: "#334155",
      tertiaryColor: "#1e293b",
      background: "#0f172a",
      mainBkg: "#1e293b",
      nodeBorder: "#475569",
      clusterBkg: "#334155",
      titleColor: "#c4b5fd",
      edgeLabelBackground: "#1e293b",
      nodeTextColor: "#e2e8f0",
      textColor: "#e2e8f0",
      fontFamily: "system-ui, sans-serif",
    },
    securityLevel: "loose",
    mindmap: {
      padding: 16,
      maxNodeWidth: 200,
    },
  });
  mermaidInitialized = true;
}

export function MemoryDiagram({ onRefreshStats }: MemoryDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const renderIdRef = useRef(0);

  const fetchAndRender = useCallback(async () => {
    if (!containerRef.current) return;
    setStatus("loading");
    setErrorMessage(null);
    containerRef.current.innerHTML = "";
    const currentId = ++renderIdRef.current;

    try {
      const res = await fetch(`${BACKEND_URL}/memory-diagram`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `Request failed (${res.status})`);
      }
      const { mermaid: code } = (await res.json()) as { mermaid: string };
      if (currentId !== renderIdRef.current) return;

      if (!code || !code.trim()) {
        setStatus("success");
        if (containerRef.current) {
          containerRef.current.innerHTML = '<p class="text-slate-500 text-sm text-center py-8">No diagram generated.</p>';
        }
        return;
      }

      initMermaid();
      const id = `mermaid-memory-${Date.now()}-${currentId}`;
      const { svg, bindFunctions } = await mermaid.render(id, code.trim());
      if (currentId !== renderIdRef.current) return;

      if (containerRef.current) {
        containerRef.current.innerHTML = svg;
        const svgEl = containerRef.current.querySelector("svg");
        if (svgEl) {
          svgEl.setAttribute("width", "100%");
          svgEl.setAttribute("height", "100%");
          svgEl.style.maxWidth = "100%";
          svgEl.style.height = "auto";
          bindFunctions?.(containerRef.current);
        }
        setStatus("success");
      }
      onRefreshStats?.();
    } catch (err) {
      if (currentId !== renderIdRef.current) return;
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Failed to load diagram");
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    }
  }, [onRefreshStats]);

  useEffect(() => {
    fetchAndRender();
  }, [fetchAndRender]);

  return (
    <div className="rounded-xl bg-slate-900/50 border border-slate-700/50 overflow-hidden flex flex-col min-h-0">
      <div className="flex items-center justify-between gap-2 p-3 border-b border-slate-700/50 flex-shrink-0">
        <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
          Your memory map
        </h3>
        <button
          type="button"
          onClick={fetchAndRender}
          disabled={status === "loading"}
          className="px-3 py-1.5 rounded-lg bg-slate-700/50 text-slate-400 text-xs font-medium hover:bg-slate-600/50 disabled:opacity-50 transition-colors"
        >
          {status === "loading" ? "Generating…" : "Refresh"}
        </button>
      </div>
      <div className="flex-1 min-h-[280px] overflow-auto p-4 flex items-center justify-center">
        {status === "loading" && (
          <p className="text-sm text-slate-500">Building your memory map…</p>
        )}
        {status === "error" && errorMessage && (
          <p className="text-sm text-red-400/90 text-center px-4">{errorMessage}</p>
        )}
        <div
          ref={containerRef}
          className="mermaid-container relative w-full min-h-[240px] flex items-center justify-center [&_svg]:max-w-full [&_svg]:h-auto"
          aria-live="polite"
          aria-busy={status === "loading"}
        />
      </div>
    </div>
  );
}
