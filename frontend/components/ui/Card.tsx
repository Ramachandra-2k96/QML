import { cn } from "@/lib/utils";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  glow?: "emerald" | "violet" | "none";
}

export function Card({ children, className, glow = "none" }: CardProps) {
  const glowMap = {
    emerald: "glow-emerald",
    violet: "glow-violet",
    none: "",
  };
  return (
    <div
      className={cn(
        "card p-6",
        glowMap[glow],
        className
      )}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
}

export function CardHeader({ title, description, icon }: CardHeaderProps) {
  return (
    <div className="mb-6 flex items-start gap-4">
      {icon && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/[0.05] ring-1 ring-white/10">
          {icon}
        </div>
      )}
      <div>
        <h2 className="text-base font-semibold text-white">{title}</h2>
        {description && (
          <p className="mt-0.5 text-sm text-slate-400">{description}</p>
        )}
      </div>
    </div>
  );
}
