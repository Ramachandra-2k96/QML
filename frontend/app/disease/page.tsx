"use client";
import { useState, useRef, useCallback } from "react";
import Image from "next/image";
import { Leaf, Upload, X } from "lucide-react";
import { predictDisease, type DiseaseModelId, type DiseaseResult } from "@/lib/api";
import { Card, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { DiseaseResultCard } from "@/components/disease/DiseaseResultCard";
import { DiseaseModelToggle } from "@/components/disease/DiseaseModelToggle";
import { cn } from "@/lib/utils";

export default function DiseasePage() {
  const [model, setModel] = useState<DiseaseModelId>("resnet18");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DiseaseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const clear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictDisease(file, model);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Page header */}
      <div className="mb-10">
        <Badge variant="violet" className="mb-3">
          <Leaf className="h-3 w-3" />
          Classification
        </Badge>
        <h1 className="text-3xl font-bold text-white sm:text-4xl">
          Plant Disease Detection
        </h1>
        <p className="mt-2 text-sm text-slate-400 max-w-xl">
          Upload a leaf image to identify disease across 38 PlantVillage
          categories using ResNet-18 or MobileNet-V3-Small.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_380px]">
        {/* ── Upload form ── */}
        <form onSubmit={onSubmit} className="space-y-6">
          <DiseaseModelToggle value={model} onChange={setModel} />

          <Card>
            <CardHeader
              title="Upload Leaf Image"
              description="Supported: jpg, jpeg, png, webp"
              icon={<Upload className="h-5 w-5 text-violet-400" />}
            />

            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => !file && inputRef.current?.click()}
              className={cn(
                "relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-colors",
                dragging
                  ? "border-violet-500/60 bg-violet-500/[0.06]"
                  : file
                  ? "border-violet-500/30 bg-violet-500/[0.03]"
                  : "border-white/10 hover:border-white/20"
              )}
            >
              {preview ? (
                <>
                  <Image
                    src={preview}
                    alt="Preview"
                    fill
                    className="rounded-xl object-contain p-2"
                  />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); clear(); }}
                    className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-black/60 text-white hover:bg-black/80"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </>
              ) : (
                <>
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/[0.05] ring-1 ring-white/10">
                    <Upload className="h-5 w-5 text-slate-400" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-white">
                      Drop image here or{" "}
                      <span className="text-violet-400">browse</span>
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Max 10 MB</p>
                  </div>
                </>
              )}

              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
            </div>

            {file && (
              <p className="mt-2 text-xs text-slate-500">
                {file.name} · {(file.size / 1024).toFixed(1)} KB
              </p>
            )}

            <div className="mt-6 flex justify-end">
              <Button type="submit" loading={loading} disabled={!file} size="lg" variant="primary" className="bg-violet-600 hover:bg-violet-500 active:bg-violet-700">
                {loading ? "Classifying…" : "Detect Disease"}
              </Button>
            </div>
          </Card>

          {error && (
            <div className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-400 ring-1 ring-red-500/20">
              {error}
            </div>
          )}
        </form>

        {/* ── Result ── */}
        <DiseaseResultCard result={result} loading={loading} />
      </div>
    </div>
  );
}
