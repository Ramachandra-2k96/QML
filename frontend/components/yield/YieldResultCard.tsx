import { TrendingUp, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/Card";
import type { YieldResult } from "@/lib/api";

interface Props {
  result: YieldResult | null;
  loading: boolean;
}

const modelLabels: Record<string, string> = {
  classical: "Classical NN",
  qnn_small: "QNN Small",
  qnn_large: "QNN Large (Hybrid)",
};

export function YieldResultCard({ result, loading }: Props) {
  if (loading) {
    return (
      <Card className="flex flex-col items-center justify-center gap-3 py-16">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-400" />
        <p className="text-sm text-slate-400">Running inference…</p>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="flex flex-col items-center justify-center gap-2 py-16 text-center">
        <TrendingUp className="h-10 w-10 text-slate-600" />
        <p className="text-sm font-medium text-slate-400">No prediction yet</p>
        <p className="text-xs text-slate-600">Fill in the form and click Predict</p>
      </Card>
    );
  }

  const yieldVal = result.predicted_yield_tonne_per_ha.toFixed(3);
  const logVal = result.predicted_yield_log.toFixed(4);

  return (
    <Card glow="emerald" className="flex flex-col gap-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 ring-1 ring-emerald-500/20">
          <TrendingUp className="h-5 w-5 text-emerald-400" />
        </div>
        <div>
          <p className="text-xs text-slate-400">Result</p>
          <p className="text-sm font-semibold text-white">
            {modelLabels[result.model_used] ?? result.model_used}
          </p>
        </div>
      </div>

      {/* Primary metric */}
      <div className="rounded-xl bg-emerald-500/[0.08] px-5 py-5 text-center ring-1 ring-emerald-500/20">
        <p className="text-xs uppercase tracking-widest text-emerald-400 font-semibold mb-1">
          Predicted Yield
        </p>
        <p className="text-5xl font-bold text-white">
          {yieldVal}
        </p>
        <p className="mt-1 text-sm text-slate-400">tonnes / hectare</p>
      </div>

      {/* Secondary metric */}
      <div className="flex items-center justify-between rounded-lg bg-white/[0.03] px-4 py-3 ring-1 ring-white/[0.06]">
        <p className="text-xs text-slate-400">Log-scale prediction</p>
        <p className="text-sm font-mono font-semibold text-slate-200">{logVal}</p>
      </div>
    </Card>
  );
}
