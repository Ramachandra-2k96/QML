import type { DiseaseModelId } from "@/lib/api";

const MODELS: { id: DiseaseModelId; name: string; params: string; note: string }[] = [
  { id: "resnet18",           name: "ResNet-18",          params: "11M params",  note: "Higher accuracy"  },
  { id: "mobilenet_v3_small", name: "MobileNet-V3-Small", params: "2.5M params", note: "Faster inference" },
];

interface Props {
  value: DiseaseModelId;
  onChange: (v: DiseaseModelId) => void;
}

export function DiseaseModelToggle({ value, onChange }: Props) {
  return (
    <div>
      <p style={{ fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-3)", marginBottom: "0.75rem" }}>
        Select Model
      </p>
      <div className="grid grid-cols-2 gap-3">
        {MODELS.map((m) => {
          const active = value === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onChange(m.id)}
              className="text-left rounded-xl p-4 transition-all"
              style={{
                background: active ? "rgba(139,92,246,0.1)" : "rgba(255,255,255,0.03)",
                border: active ? "1px solid rgba(139,92,246,0.35)" : "1px solid var(--border)",
              }}
            >
              <p style={{
                fontSize: "0.8rem", fontWeight: 700,
                color: active ? "var(--vi-lt)" : "var(--text)",
              }}>
                {m.name}
              </p>
              <p style={{ fontSize: "0.7rem", color: "var(--text-3)", marginTop: "0.25rem" }}>
                {m.params} · {m.note}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
