// Typed API client — built from the actual FastAPI spec in main.py
// POST /predict/yield  — JSON body
// POST /predict/disease — multipart/form-data + ?model= query param

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Exact literals accepted by the backend ──────────────────────
export type YieldModelId = "classical" | "qnn_small" | "qnn_large";
export type DiseaseModelId = "resnet18" | "mobilenet_v3_small";

// ── Request / Response shapes (mirrors Pydantic models) ─────────
export interface YieldPayload {
  crop: string;
  season: string;
  state: string;
  area: number;           // hectares
  annual_rainfall: number; // mm
  fertilizer: number;    // kg/ha
  pesticide: number;     // kg/ha
  model: YieldModelId;
}

export interface YieldResult {
  model_used: string;
  predicted_yield_log: number;
  predicted_yield_tonne_per_ha: number;
}

export interface DiseaseResult {
  model_used: string;
  predicted_class: string;  // e.g. "Apple___Apple_scab"
  confidence: number;        // 0-100 float
}

// ── Helpers ──────────────────────────────────────────────────────
async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      msg = body?.detail ?? msg;
    } catch {
      // keep default
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

// ── API calls ────────────────────────────────────────────────────
export async function predictYield(payload: YieldPayload): Promise<YieldResult> {
  const res = await fetch(`${BASE}/predict/yield`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return unwrap<YieldResult>(res);
}

// model is sent as a query param: /predict/disease?model=resnet18
// file is sent as multipart field named "file"
export async function predictDisease(
  file: File,
  model: DiseaseModelId
): Promise<DiseaseResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/predict/disease?model=${model}`, {
    method: "POST",
    body: form,
  });
  return unwrap<DiseaseResult>(res);
}
