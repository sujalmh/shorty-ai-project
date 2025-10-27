import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

type Props = {
  figure: any; // Plotly figure dict from backend
  onClose: () => void;
  title?: string;
};

// ---- helpers to normalize backend figure to Plotly-consumable shapes in the browser ----
function decodeBase64ToTypedArray(dtype: string, b64: string): number[] {
  try {
    const binStr = atob(b64);
    const len = binStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binStr.charCodeAt(i);
    const dv = new DataView(bytes.buffer);
    switch (dtype) {
      case "u1": return Array.from(bytes);
      case "i2": {
        const arr = new Int16Array(bytes.buffer);
        return Array.from(arr);
      }
      case "u2": {
        const arr = new Uint16Array(bytes.buffer);
        return Array.from(arr);
      }
      case "i4": {
        const arr = new Int32Array(bytes.buffer);
        return Array.from(arr);
      }
      case "u4": {
        const arr = new Uint32Array(bytes.buffer);
        return Array.from(arr);
      }
      case "f4": {
        const arr = new Float32Array(bytes.buffer);
        return Array.from(arr);
      }
      case "f8": {
        const arr = new Float64Array(bytes.buffer);
        return Array.from(arr);
      }
      default: {
        // fallback read as 16-bit signed
        const arr = new Int16Array(bytes.buffer);
        return Array.from(arr);
      }
    }
  } catch {
    return [];
  }
}

function parseLabelsMaybeString(v: any): any {
  if (typeof v === "string") {
    // Try to coerce strings like "['East' 'North' 'South' 'West']" into array
    const trimmed = v.trim();
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
      const jsonish = trimmed.replace(/'/g, '"').replace(/\s+/g, ", ").replace(/, \]/g, "]");
      try {
        const arr = JSON.parse(jsonish);
        if (Array.isArray(arr)) return arr;
      } catch {
        // fallback split
        const inner = trimmed.slice(1, -1).trim();
        if (inner.length === 0) return [];
        return inner.split(/[,\s]+/).map((s) => s.replace(/^"+|"+$/g, "").replace(/^'+|'+$/g, ""));
      }
    }
  }
  return v;
}

function normalizeTrace(trace: any): any {
  const t = { ...trace };
  // Normalize common array fields
  for (const key of ["x", "y", "z", "values", "labels"]) {
    const v = (t as any)[key];
    if (!v) continue;
    if (v && typeof v === "object" && "dtype" in v && "bdata" in v) {
      (t as any)[key] = decodeBase64ToTypedArray(v.dtype, v.bdata);
    } else if (key === "labels") {
      (t as any)[key] = parseLabelsMaybeString(v);
    }
  }
  return t;
}

function normalizeFigure(fig: any): { data: any[]; layout: any; config: any } {
  const data = Array.isArray(fig?.data) ? fig.data.map((tr: any) => normalizeTrace(tr)) : [];
  const titleIn = fig?.layout?.title;
  const title =
    typeof titleIn === "string"
      ? titleIn
      : titleIn && typeof titleIn === "object" && "text" in titleIn
      ? (titleIn as any).text
      : undefined;
  const layout = {
    ...fig?.layout,
    title,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    autosize: true,
    margin: { t: 36, r: 16, b: 40, l: 48, ...(fig?.layout?.margin ?? {}) },
    font: { color: "currentColor", ...(fig?.layout?.font ?? {}) },
  };
  const config = { responsive: true, displaylogo: false, ...(fig?.config ?? {}) };
  return { data, layout, config };
}

export default function PlotViewer({ figure, onClose, title }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || !figure) return;
    const el = ref.current;

    const { data, layout, config } = normalizeFigure(figure);
    const finalLayout = {
      ...layout,
      title: title ?? layout.title ?? "",
    };

    Plotly.newPlot(el, data, finalLayout, config);

    const handleResize = () => {
      Plotly.Plots.resize(el);
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      Plotly.purge(el);
    };
  }, [figure, title]);

  const headerTitle =
    title ??
    (typeof figure?.layout?.title === "string"
      ? figure?.layout?.title
      : figure?.layout?.title?.text) ??
    "Chart";

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-lg border border-gray-200/60 shadow-md overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200/60 bg-white/70">
        <span className="text-sm font-semibold text-gray-700">{headerTitle}</span>
        <button
          className="btn-secondary"
          onClick={onClose}
        >
          Close
        </button>
      </div>
      <div className="p-4" style={{ width: "100%", height: 360 }}>
        <div ref={ref} style={{ width: "100%", height: "100%" }} />
      </div>
    </div>
  );
}
