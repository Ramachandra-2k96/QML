"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV = [
  { href: "/",        label: "Home" },
  { href: "/yield",   label: "Yield Prediction" },
  { href: "/disease", label: "Disease Detection" },
];

export function Navbar() {
  const path = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      <header
        style={{ background: "rgba(8,10,15,0.85)", backdropFilter: "blur(20px)" }}
        className="fixed inset-x-0 top-0 z-50 border-b border-white/[0.06]"
      >
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-8 px-5 lg:px-8">

          {/* Logo */}
          <Link href="/" className="flex shrink-0 items-center gap-3">
            <span
              className="flex h-8 w-8 items-center justify-center rounded-xl text-sm font-black text-white"
              style={{ background: "linear-gradient(135deg,#10b981,#059669)" }}
            >
              A
            </span>
            <span className="text-sm font-bold tracking-tight text-white">
              Agri<span className="text-emerald-400">QML</span>
            </span>
          </Link>

          {/* Desktop nav — centered */}
          <nav className="hidden md:flex items-center gap-1">
            {NAV.map(({ href, label }) => {
              const active = path === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                    active
                      ? "bg-white/10 text-white"
                      : "text-slate-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  {label}
                </Link>
              );
            })}
          </nav>

          {/* CTA */}
          <div className="hidden md:block shrink-0">
            <Link
              href="/yield"
              className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-400 transition-colors"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-white/5"
            onClick={() => setOpen(!open)}
            aria-label="Menu"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="h-5 w-5">
              {open
                ? <><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></>
                : <><line x1="3" y1="7" x2="21" y2="7" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="17" x2="21" y2="17" /></>
              }
            </svg>
          </button>
        </div>
      </header>

      {/* Mobile menu */}
      {open && (
        <div
          className="fixed inset-x-0 top-16 z-40 border-b border-white/[0.06] md:hidden"
          style={{ background: "rgba(8,10,15,0.97)", backdropFilter: "blur(20px)" }}
        >
          <div className="mx-auto max-w-7xl flex flex-col gap-1 px-5 py-4">
            {NAV.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={`rounded-lg px-4 py-3 text-sm font-medium transition-colors ${
                  path === href ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                {label}
              </Link>
            ))}
            <Link
              href="/yield"
              onClick={() => setOpen(false)}
              className="mt-2 rounded-lg bg-emerald-500 py-3 text-center text-sm font-semibold text-white"
            >
              Get Started
            </Link>
          </div>
        </div>
      )}

      {/* Push content below fixed header */}
      <div className="h-16" />
    </>
  );
}
