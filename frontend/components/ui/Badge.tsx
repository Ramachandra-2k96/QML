import { cn } from "@/lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "emerald" | "violet" | "slate";
  className?: string;
}

export function Badge({ children, variant = "slate", className }: BadgeProps) {
  const vars = {
    emerald: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/20",
    violet: "bg-violet-500/10 text-violet-400 ring-violet-500/20",
    slate: "bg-white/[0.06] text-slate-400 ring-white/10",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ring-1",
        vars[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
