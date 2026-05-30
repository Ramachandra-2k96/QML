import { Brain, Cpu, Atom } from "lucide-react";
import { cn } from "@/lib/utils";
import type { YieldModelId } from "@/lib/api";

const models: { id: YieldModelId; label: string; sub: string; icon: React.ReactNode; badge: string }[] = [
  {
    id: "classical",
    label: "Classical NN",
    sub: "3.3M params · ResBlock MLP",
    icon: <Cpu className="h-5 w-5" />,
    badge: "Baseline",
  },
  {
    id: "qnn_small",
    label: "QNN Small",
    sub: "408 params · 7-qubit VQC",
    icon: <Atom className="h-5 w-5" />,
    badge: "Quantum",
  },
  {
    id: "qnn_large",
    label: "QNN Large",
    sub: "R² 0.91 · Quantum + 3.3M",
    icon: <Brain className="h-5 w-5" />,
    badge: "Hybrid ★",
  },
];

interface Props {
  value: YieldModelId;
  onChange: (v: YieldModelId) => void;
}

export function ModelSelector({ value, onChange }: Props) {
  return (
    <div>
      <p className="mb-2 text-xs font-medium text-slate-300">Select Model</p>
      <div className="grid grid-cols-3 gap-3">
        {models.map((m) => {
          const active = value === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onChange(m.id)}
              className={cn(
                "relative flex flex-col items-start gap-2 rounded-xl p-3.5 text-left transition-all ring-1",
                active
                  ? "bg-emerald-500/10 ring-emerald-500/40"
                  : "bg-white/[0.03] ring-white/10 hover:bg-white/[0.06]"
              )}
            >
              <div
                className={cn(
                  "flex h-8 w-8 items-center justify-center rounded-lg",
                  active ? "bg-emerald-500/20 text-emerald-400" : "bg-white/[0.06] text-slate-400"
                )}
              >
                {m.icon}
              </div>
              <div>
                <p className={cn("text-xs font-semibold", active ? "text-white" : "text-slate-300")}>
                  {m.label}
                </p>
                <p className="text-[10px] text-slate-500 leading-tight mt-0.5">{m.sub}</p>
              </div>
              <span
                className={cn(
                  "absolute right-2 top-2 rounded-full px-1.5 py-0.5 text-[9px] font-semibold",
                  active ? "bg-emerald-500/20 text-emerald-400" : "bg-white/[0.06] text-slate-500"
                )}
              >
                {m.badge}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
