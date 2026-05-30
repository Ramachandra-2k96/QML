"use client";
import { useState } from "react";
import { BarChart3, Atom } from "lucide-react";
import { predictYield, type YieldPayload, type YieldResult, type YieldModelId } from "@/lib/api";
import { Card, CardHeader } from "@/components/ui/Card";
import { Input, Select } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { YieldResultCard } from "@/components/yield/YieldResultCard";
import { ModelSelector } from "@/components/yield/ModelSelector";

// ── Exact values from crop_yield.csv ──────────────────────────
const CROPS = [
  "Arecanut","Arhar/Tur","Bajra","Banana","Barley","Black pepper","Cardamom",
  "Cashewnut","Castor seed","Coconut","Coriander","Cotton(lint)","Cowpea(Lobia)",
  "Dry chillies","Garlic","Ginger","Gram","Groundnut","Guar seed","Horse-gram",
  "Jowar","Jute","Khesari","Linseed","Maize","Masoor","Mesta","Moong(Green Gram)",
  "Moth","Niger seed","Oilseeds total","Onion","Other  Rabi pulses","Other Cereals",
  "Other Kharif pulses","Other Summer Pulses","Peas & beans (Pulses)","Potato",
  "Ragi","Rapeseed &Mustard","Rice","Safflower","Sannhamp","Sesamum","Small millets",
  "Soyabean","Sugarcane","Sunflower","Sweet potato","Tapioca","Tobacco","Turmeric",
  "Urad","Wheat","other oilseeds",
];

const SEASONS = ["Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"];

const STATES = [
  "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Delhi",
  "Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand",
  "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya",
  "Mizoram","Nagaland","Odisha","Puducherry","Punjab","Sikkim","Tamil Nadu",
  "Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
];

const defaultForm: Omit<YieldPayload, "model"> = {
  crop: "Rice",
  season: "Kharif",
  state: "Punjab",
  area: 100,
  annual_rainfall: 800,
  fertilizer: 120,
  pesticide: 5,
};

export default function YieldPage() {
  const [form, setForm] = useState(defaultForm);
  const [model, setModel] = useState<YieldModelId>("classical");
  const [result, setResult] = useState<YieldResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set =
    (key: keyof typeof form) =>
    (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      const val =
        e.target.type === "number" ? parseFloat(e.target.value) || 0 : e.target.value;
      setForm((f) => ({ ...f, [key]: val }));
    };

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictYield({ ...form, model });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Page header */}
      <div className="mb-10">
        <Badge variant="emerald" className="mb-3">
          <BarChart3 className="h-3 w-3" />
          Regression
        </Badge>
        <h1 className="text-3xl font-bold text-white sm:text-4xl">
          Crop Yield Prediction
        </h1>
        <p className="mt-2 max-w-xl text-sm text-slate-400">
          Enter field parameters to predict yield in tonnes/hectare. Choose
          between Classical NN, compact Quantum, or the full Quantum-Hybrid model.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_420px]">
        {/* ── Form ── */}
        <form onSubmit={onSubmit} className="space-y-6">
          <ModelSelector value={model} onChange={setModel} />

          <Card>
            <CardHeader
              title="Field Parameters"
              description="Select from the exact 55 crops, 6 seasons, and 30 states present in the training data."
              icon={<BarChart3 className="h-5 w-5 text-emerald-400" />}
            />
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Crop" value={form.crop} onChange={set("crop")}>
                {CROPS.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Select>
              <Select label="Season" value={form.season} onChange={set("season")}>
                {SEASONS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </Select>
              <Select label="State" value={form.state} onChange={set("state")}>
                {STATES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </Select>
              <Input
                label="Area (hectares)"
                type="number"
                min={0}
                step={0.1}
                value={form.area}
                onChange={set("area")}
              />
              <Input
                label="Annual Rainfall (mm)"
                type="number"
                min={0}
                step={1}
                value={form.annual_rainfall}
                onChange={set("annual_rainfall")}
              />
              <Input
                label="Fertilizer (kg/ha)"
                type="number"
                min={0}
                step={0.1}
                value={form.fertilizer}
                onChange={set("fertilizer")}
              />
              <Input
                label="Pesticide (kg/ha)"
                type="number"
                min={0}
                step={0.01}
                value={form.pesticide}
                onChange={set("pesticide")}
              />
            </div>

            <div className="mt-6 flex justify-end">
              <Button type="submit" loading={loading} size="lg">
                {loading ? "Predicting…" : "Predict Yield"}
              </Button>
            </div>
          </Card>

          {error && (
            <div className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-400 ring-1 ring-red-500/20">
              {error}
            </div>
          )}
        </form>

        {/* ── Result + info ── */}
        <div className="flex flex-col gap-5">
          <YieldResultCard result={result} loading={loading} />
          <Card>
            <div className="flex items-start gap-3">
              <Atom className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
              <div>
                <p className="text-sm font-medium text-white">Model details</p>
                <ul className="mt-2 space-y-1.5 text-xs text-slate-400">
                  <li>
                    <span className="font-medium text-slate-300">Classical NN</span>
                    {" "}— 6-block Residual MLP, 3.3M params.
                  </li>
                  <li>
                    <span className="font-medium text-slate-300">QNN Small</span>
                    {" "}— Pre-layer → 7-qubit VQC → head. Only 408 params.
                  </li>
                  <li>
                    <span className="font-medium text-slate-300">QNN Large</span>
                    {" "}— Frozen VQC features → 3.3M classical head. R² 0.91.
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
