import { type DiseaseResult } from "@/lib/api";
import { CheckCircle, AlertTriangle, Loader } from "lucide-react";

interface Props {
  result: DiseaseResult | null;
  loading: boolean;
}

function formatClass(raw: string): { plant: string; disease: string } {
  const parts = raw.split("___");
  if (parts.length === 2) {
    return {
      plant: parts[0].replace(/_/g, " "),
      disease: parts[1].replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    };
  }
  return { plant: "", disease: raw.replace(/_/g, " ") };
}

export function DiseaseResultCard({ result, loading }: Props) {
  if (loading) {
    return (
      <div className="card flex flex-col items-center justify-center gap-3 py-20">
        <Loader className="h-8 w-8 spinner" style={{ color: "var(--vi)" }} />
        <p style={{ color: "var(--text-2)", fontSize: "0.875rem" }}>Running classifier…</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card flex flex-col items-center justify-center gap-3 py-20 text-center px-6">
        <svg className="h-12 w-12" style={{ color: "var(--text-4)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2}
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <p style={{ color: "var(--text-3)", fontSize: "0.875rem", fontWeight: 600 }}>
          No result yet
        </p>
        <p style={{ color: "var(--text-4)", fontSize: "0.75rem" }}>
          Upload a leaf image and click Detect
        </p>
      </div>
    );
  }

  const isHealthy = result.predicted_class.toLowerCase().includes("healthy");
  const { plant, disease } = formatClass(result.predicted_class);
  const accentColor = isHealthy ? "var(--em)" : "#f97316";
  const accentBg = isHealthy ? "rgba(16,185,129,0.08)" : "rgba(249,115,22,0.08)";
  const accentBorder = isHealthy ? "rgba(16,185,129,0.2)" : "rgba(249,115,22,0.2)";

  const barColor =
    result.confidence > 80 ? "var(--em)" :
    result.confidence > 55 ? "#eab308" : "#ef4444";

  return (
    <div className="card p-7 flex flex-col gap-6" style={isHealthy ? { boxShadow: "0 0 40px rgba(16,185,129,0.08)" } : {}}>

      {/* Header */}
      <div className="flex items-center justify-between">
        <span style={{ color: "var(--text-3)", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Result
        </span>
        <span className={isHealthy ? "badge-em" : "badge-vi"}>
          {result.model_used}
        </span>
      </div>

      {/* Main result */}
      <div
        className="rounded-2xl p-6 text-center"
        style={{ background: accentBg, border: `1px solid ${accentBorder}` }}
      >
        <div className="flex justify-center mb-3">
          {isHealthy
            ? <CheckCircle className="h-8 w-8" style={{ color: accentColor }} />
            : <AlertTriangle className="h-8 w-8" style={{ color: accentColor }} />
          }
        </div>
        <p className="text-xs font-bold uppercase tracking-widest mb-2" style={{ color: accentColor }}>
          {isHealthy ? "Healthy Plant" : "Disease Detected"}
        </p>
        {plant && (
          <p className="text-sm mb-1" style={{ color: "var(--text-2)" }}>{plant}</p>
        )}
        <p className="text-xl font-bold" style={{ color: "var(--text)" }}>{disease}</p>
      </div>

      {/* Confidence */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-xs" style={{ color: "var(--text-3)" }}>Confidence</p>
          <p className="font-mono text-sm font-bold" style={{ color: "var(--text)" }}>
            {result.confidence.toFixed(1)}%
          </p>
        </div>
        <div className="h-2 w-full rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
          <div
            className="h-full rounded-full"
            style={{ width: `${result.confidence}%`, background: barColor, transition: "width 0.6s ease" }}
          />
        </div>
        <p style={{ fontSize: "0.7rem", color: "var(--text-4)" }}>
          {result.confidence > 85
            ? "High confidence — reliable prediction"
            : result.confidence > 60
            ? "Moderate confidence — consider manual verification"
            : "Low confidence — try a cleaner, well-lit image"}
        </p>
      </div>
    </div>
  );
}
