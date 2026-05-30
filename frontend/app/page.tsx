import Link from "next/link";
import { ArrowRight } from "lucide-react";

const STATS = [
  { n: "0.91",  label: "R² Score",      note: "Quantum Hybrid" },
  { n: "55",    label: "Crop Types",     note: "Indian dataset" },
  { n: "38",    label: "Disease Classes",note: "PlantVillage" },
  { n: "408",   label: "Quantum Params", note: "vs 3.3M classical" },
];

const FEATURES = [
  {
    accent: "#10b981",
    tag: "Regression",
    title: "Crop Yield Prediction",
    body: "Three model tiers — Classical NN (3.3M params), compact 7-qubit QNN (408 params), and Quantum-Hybrid (R² 0.91) — for principled ablation comparison across 55 crops, 6 seasons, and 30 Indian states.",
    href: "/yield",
    cta: "Predict yield",
  },
  {
    accent: "#8b5cf6",
    tag: "Classification",
    title: "Plant Disease Detection",
    body: "ResNet-18 (11M) and MobileNet-V3-Small (2.5M) fine-tuned on 38 PlantVillage disease categories. Upload any leaf image for instant classification with confidence scoring.",
    href: "/disease",
    cta: "Detect disease",
  },
  {
    accent: "#10b981",
    tag: "Architecture",
    title: "Quantum-Hybrid Network",
    body: "7-qubit Variational Quantum Circuit using StronglyEntanglingLayers. Trained end-to-end for 30 epochs, then frozen — quantum features extracted and fed into a 3.3M-param Residual head.",
    href: "/yield",
    cta: "Try it",
  },
];

export default function HomePage() {
  return (
    <main>

      {/* ── HERO ── */}
      <section className="relative overflow-hidden">
        <div className="dot-grid pointer-events-none absolute inset-0" />

        {/* Glow blobs */}
        <div
          className="pointer-events-none absolute"
          style={{
            top: "-15%", left: "50%", transform: "translateX(-50%)",
            width: 900, height: 700,
            background: "radial-gradient(ellipse,rgba(16,185,129,.12) 0%,transparent 70%)",
            filter: "blur(40px)",
          }}
        />
        <div
          className="pointer-events-none absolute"
          style={{
            bottom: 0, right: "-10%",
            width: 500, height: 500,
            background: "radial-gradient(ellipse,rgba(139,92,246,.1) 0%,transparent 70%)",
            filter: "blur(40px)",
          }}
        />

        <div className="relative mx-auto max-w-7xl px-5 pb-28 pt-24 lg:px-8 lg:pt-36">
          <div className="mx-auto max-w-3xl">

            {/* Eyebrow */}
            <div
              className="mb-8 inline-flex items-center gap-2 rounded-full px-3.5 py-1 text-xs font-semibold text-emerald-400"
              style={{ background: "rgba(16,185,129,.1)", border: "1px solid rgba(16,185,129,.2)" }}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Quantum Machine Learning for Agriculture
            </div>

            {/* Headline */}
            <h1 className="g-hero text-5xl font-extrabold leading-[1.08] tracking-tight sm:text-6xl xl:text-7xl">
              Smarter farming
              <br />with Quantum AI
            </h1>

            <p className="mt-7 max-w-xl text-base leading-relaxed text-slate-400 sm:text-lg">
              Hybrid Quantum-Classical Neural Networks for crop yield prediction
              and plant disease detection — real models, real data, real results.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                href="/yield"
                className="inline-flex items-center gap-2.5 rounded-xl px-7 py-3.5 text-sm font-semibold text-white"
                style={{
                  background: "linear-gradient(135deg,#10b981,#059669)",
                  boxShadow: "0 0 32px rgba(16,185,129,.25)",
                }}
              >
                Predict Yield <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/disease"
                className="inline-flex items-center gap-2.5 rounded-xl px-7 py-3.5 text-sm font-semibold text-white"
                style={{ background: "rgba(255,255,255,.06)", border: "1px solid rgba(255,255,255,.1)" }}
              >
                Detect Disease
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── STATS STRIP ── */}
      <section
        style={{ borderTop: "1px solid rgba(255,255,255,.06)", borderBottom: "1px solid rgba(255,255,255,.06)" }}
        className="bg-white/[.02]"
      >
        <div className="mx-auto max-w-7xl px-5 py-10 lg:px-8">
          <dl className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            {STATS.map(({ n, label, note }) => (
              <div key={label} className="text-center">
                <dt className="g-em text-4xl font-extrabold">{n}</dt>
                <dd className="mt-1 text-sm font-semibold text-white">{label}</dd>
                <dd className="text-xs text-slate-500">{note}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section className="mx-auto max-w-7xl px-5 py-24 lg:px-8">
        <h2 className="mb-3 text-2xl font-bold text-white sm:text-3xl">What's inside</h2>
        <p className="mb-14 max-w-lg text-sm text-slate-400 leading-relaxed">
          Two ML tasks, five trained models, one FastAPI backend. Every number
          is backed by training logs, not speculation.
        </p>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="group flex flex-col gap-5 rounded-2xl p-7 transition-colors"
              style={{
                background: "rgba(14,17,24,1)",
                border: "1px solid rgba(255,255,255,.06)",
              }}
            >
              {/* Tag */}
              <span
                className="self-start rounded-full px-3 py-1 text-[11px] font-semibold"
                style={{
                  background: `${f.accent}18`,
                  color: f.accent,
                  border: `1px solid ${f.accent}33`,
                }}
              >
                {f.tag}
              </span>

              <div>
                <h3 className="mb-2 text-base font-bold text-white">{f.title}</h3>
                <p className="text-sm leading-relaxed text-slate-400">{f.body}</p>
              </div>

              <Link
                href={f.href}
                className="mt-auto inline-flex items-center gap-1.5 text-sm font-semibold transition-colors"
                style={{ color: f.accent }}
              >
                {f.cta} <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* ── BOTTOM CTA ── */}
      <section className="mx-auto max-w-7xl px-5 pb-24 lg:px-8">
        <div
          className="relative overflow-hidden rounded-2xl px-8 py-16 text-center"
          style={{
            background: "linear-gradient(135deg, rgba(16,185,129,.08) 0%, rgba(14,17,24,1) 50%, rgba(139,92,246,.08) 100%)",
            border: "1px solid rgba(255,255,255,.07)",
          }}
        >
          <h2 className="text-2xl font-bold text-white sm:text-3xl">Ready to run a prediction?</h2>
          <p className="mx-auto mt-3 max-w-sm text-sm text-slate-400">
            No login, no setup. Just fill the form and see the quantum model work.
          </p>
          <div className="mt-8 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
            <Link
              href="/yield"
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-6 py-3 text-sm font-semibold text-white hover:bg-emerald-400 transition-colors"
            >
              Yield Prediction <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/disease"
              className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-white transition-colors"
              style={{ background: "rgba(255,255,255,.06)", border: "1px solid rgba(255,255,255,.1)" }}
            >
              Disease Detection
            </Link>
          </div>
        </div>
      </section>

    </main>
  );
}
