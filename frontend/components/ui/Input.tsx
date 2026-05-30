import { cn } from "@/lib/utils";
import { forwardRef } from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, hint, error, className, ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-xs font-medium text-slate-300">{label}</label>
      )}
      <input
        ref={ref}
        className={cn(
          "w-full rounded-lg bg-white/[0.05] px-3 py-2 text-sm text-white",
          "ring-1 ring-white/10 placeholder:text-slate-500",
          "transition focus:outline-none focus:ring-emerald-500/50",
          error && "ring-red-500/50",
          className
        )}
        {...props}
      />
      {hint && !error && <p className="text-[11px] text-slate-500">{hint}</p>}
      {error && <p className="text-[11px] text-red-400">{error}</p>}
    </div>
  )
);
Input.displayName = "Input";

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, error, className, children, ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-xs font-medium text-slate-300">{label}</label>
      )}
      <select
        ref={ref}
        className={cn(
          "w-full rounded-lg bg-[#181d28] px-3 py-2 text-sm text-white",
          "ring-1 ring-white/10 transition focus:outline-none focus:ring-emerald-500/50",
          "appearance-none cursor-pointer",
          error && "ring-red-500/50",
          className
        )}
        {...props}
      >
        {children}
      </select>
      {error && <p className="text-[11px] text-red-400">{error}</p>}
    </div>
  )
);
Select.displayName = "Select";
