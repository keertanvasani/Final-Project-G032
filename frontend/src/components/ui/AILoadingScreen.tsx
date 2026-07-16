import React, { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { CheckCircle2, Cpu, Layers, Sparkles, Zap, ScanSearch } from "lucide-react"

interface AILoadingScreenProps {
  onComplete?: () => void
}

const PIPELINE_STEPS = [
  {
    id: 1,
    name: "YOLOv8-n Detection",
    label: "Object Detector — Nano",
    icon: ScanSearch,
    durationMs: 900,
  },
  {
    id: 2,
    name: "YOLOv8-s Verification",
    label: "Object Detector — Small",
    icon: Cpu,
    durationMs: 800,
  },
  {
    id: 3,
    name: "Weighted Box Fusion",
    label: "Ensemble Box Merger",
    icon: Layers,
    durationMs: 700,
  },
  {
    id: 4,
    name: "Species Classification",
    label: "CNN Categorizer",
    icon: Zap,
    durationMs: 600,
  },
  {
    id: 5,
    name: "Generating Final Prediction",
    label: "Output Labels & Bounding Boxes",
    icon: Sparkles,
    durationMs: 500,
  },
]

const TOTAL_DURATION_MS = PIPELINE_STEPS.reduce((s, p) => s + p.durationMs, 0)

export const AILoadingScreen: React.FC<AILoadingScreenProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0) // 0 = none complete yet
  const [progress, setProgress] = useState(0)

  /* ── Step sequencer ── */
  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = []

    PIPELINE_STEPS.forEach((step, idx) => {
      const delay = PIPELINE_STEPS.slice(0, idx).reduce((s, p) => s + p.durationMs, 0)
      timers.push(
        setTimeout(() => {
          setCurrentStep(idx + 1)
        }, delay + step.durationMs)
      )
    })

    /* Mark fully done and notify parent */
    timers.push(
      setTimeout(() => {
        setCurrentStep(PIPELINE_STEPS.length + 1) // "all done"
        onComplete?.()
      }, TOTAL_DURATION_MS + 300)
    )

    return () => timers.forEach(clearTimeout)
  }, [onComplete])

  /* ── Smooth progress bar via rAF ── */
  useEffect(() => {
    let start: number | null = null
    let rafId: number

    const tick = (ts: number) => {
      if (!start) start = ts
      const pct = Math.min(((ts - start) / TOTAL_DURATION_MS) * 100, 100)
      setProgress(pct)
      if (pct < 100) rafId = requestAnimationFrame(tick)
    }

    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [])


  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-navy-950/85 backdrop-blur-md rounded-2xl p-6 sm:p-10">
      
      {/* ── Header ── */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8 space-y-1"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-xs font-bold text-cyan-400 mb-3">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          Running Ensemble Detection
        </div>
        <h3 className="text-lg font-extrabold text-white tracking-tight">
          Analyzing Your Image
        </h3>
        <p className="text-xs text-muted-foreground/80">
          Multi-model inference pipeline in progress
        </p>
      </motion.div>

      {/* ── Pipeline Steps ── */}
      <div className="w-full max-w-md space-y-0">
        {PIPELINE_STEPS.map((step, idx) => {
          const isDone = currentStep > idx + 1
          const isActive = currentStep === idx + 1

          return (
            <React.Fragment key={step.id}>
              {/* Step row */}
              <motion.div
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.08, duration: 0.35 }}
                className={cn(
                  "flex items-center gap-4 px-4 py-3 rounded-xl border transition-all duration-500",
                  isDone
                    ? "bg-teal-500/5 border-teal-500/20"
                    : isActive
                    ? "bg-cyan-500/10 border-cyan-500/30 shadow-glow-sm/10"
                    : "bg-navy-950/30 border-navy-800/40 opacity-40"
                )}
              >
                {/* Step number / icon / checkmark */}
                <div className="relative shrink-0 w-9 h-9">
                  <AnimatePresence mode="wait">
                    {isDone ? (
                      <motion.div
                        key="done"
                        initial={{ scale: 0.4, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.4, opacity: 0 }}
                        transition={{ type: "spring", stiffness: 300, damping: 18 }}
                        className="w-full h-full rounded-xl bg-teal-500/15 border border-teal-500/30 flex items-center justify-center"
                      >
                        <CheckCircle2 className="w-4.5 h-4.5 text-teal-400" />
                      </motion.div>
                    ) : isActive ? (
                      <motion.div
                        key="active"
                        initial={{ scale: 0.7, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.7, opacity: 0 }}
                        className="w-full h-full rounded-xl bg-cyan-500/15 border border-cyan-500/35 flex items-center justify-center"
                      >
                        {/* Dual-ring pulse */}
                        <span className="absolute inset-0 rounded-xl border border-cyan-400/40 animate-ping" />
                        <step.icon className="w-4 h-4 text-cyan-400" />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="pending"
                        className="w-full h-full rounded-xl bg-navy-900/60 border border-navy-800/60 flex items-center justify-center"
                      >
                        <step.icon className="w-4 h-4 text-muted-foreground/50" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Step label */}
                <div className="flex-1 min-w-0">
                  <div className={cn(
                    "text-xs font-bold truncate transition-colors duration-300",
                    isDone ? "text-teal-400" : isActive ? "text-white" : "text-muted-foreground/50"
                  )}>
                    Step {step.id} — {step.name}
                  </div>
                  <div className="text-[10px] text-muted-foreground/60 truncate mt-0.5">
                    {step.label}
                  </div>
                </div>

                {/* Right status text */}
                <div className="shrink-0 text-[10px] font-bold">
                  <AnimatePresence mode="wait">
                    {isDone && (
                      <motion.span
                        key="done-label"
                        initial={{ opacity: 0, x: 6 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0 }}
                        className="text-teal-400"
                      >
                        ✓ Done
                      </motion.span>
                    )}
                    {isActive && (
                      <motion.span
                        key="active-label"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0.4, 1, 0.4] }}
                        transition={{ repeat: Infinity, duration: 1.4 }}
                        className="text-cyan-400"
                      >
                        Processing...
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>

              {/* Glowing vertical connector — not after last step */}
              {idx < PIPELINE_STEPS.length - 1 && (
                <div className="flex items-center justify-center my-0">
                  <div className="flex flex-col items-center gap-[2px] py-1">
                    {[0, 1, 2].map((dot) => (
                      <motion.div
                        key={dot}
                        className="w-0.5 h-1.5 rounded-full"
                        animate={{
                          backgroundColor: currentStep > idx + 1
                            ? "rgba(45, 212, 191, 0.8)"   // teal — done
                            : currentStep === idx + 1
                            ? "rgba(34, 211, 238, 0.8)"   // cyan — active
                            : "rgba(34, 211, 238, 0.15)", // muted — pending
                          boxShadow: currentStep > idx + 1
                            ? "0 0 6px rgba(45, 212, 191, 0.5)"
                            : currentStep === idx + 1
                            ? "0 0 8px rgba(34, 211, 238, 0.6)"
                            : "none",
                        }}
                        transition={{ delay: dot * 0.1, duration: 0.4 }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* ── Progress Bar ── */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="w-full max-w-md mt-8 space-y-2"
      >
        <div className="flex justify-between text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">
          <span>Pipeline Progress</span>
          <span className="text-cyan-400">{Math.round(progress)}%</span>
        </div>
        <div className="w-full h-2 bg-navy-950 rounded-full overflow-hidden border border-navy-800/60">
          <motion.div
            className="h-full rounded-full relative overflow-hidden"
            style={{
              width: `${progress}%`,
              background: "linear-gradient(90deg, #0EA5E9, #06B6D4, #14B8A6)",
              boxShadow: "0 0 12px rgba(34, 211, 238, 0.5)",
            }}
            transition={{ ease: "linear" }}
          >
            {/* Shimmer sweep */}
            <div
              className="absolute inset-0 opacity-40"
              style={{
                background:
                  "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%)",
                backgroundSize: "200px 100%",
                animation: "shimmer 1.5s linear infinite",
              }}
            />
          </motion.div>
        </div>
      </motion.div>
    </div>
  )
}

/* tiny cn helper inline to avoid import if not using @/lib/utils */
function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(" ")
}

export default AILoadingScreen
