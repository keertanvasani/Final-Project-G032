import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import {
  CheckCircle2,
  Download,
  UploadCloud,
  Share2,
  ArrowLeft,
  ShieldCheck,
  Cpu,
  Layers,
  Sparkles,
  Clock,
  Monitor,
  Image as ImageIcon,
  Calendar,
  Database,
  ScanSearch,
  Zap
} from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import originalImg from "@/assets/underwater.png"
import annotatedImg from "@/assets/sample_detection.png"

/* ─────────────────────────────────────────────────────────────────────────
   Mock Data
───────────────────────────────────────────────────────────────────────── */
const DETECTIONS: {
  id: string
  species: string
  scientific: string
  detectorConf: number
  classifierConf: number
  combinedConf: number
  status: string
  highlight: boolean
}[] = []

const PIPELINE_CHECKS = [
  { name: "YOLOv8-n", result: "--", icon: ScanSearch },
  { name: "YOLOv8-s", result: "--", icon: Cpu },
  { name: "Weighted Box Fusion", result: "--", icon: Layers },
  { name: "Classifier", result: "--", icon: Sparkles },
  { name: "Final Decision", result: "--", icon: CheckCircle2 },
]

const STATS = [
  { label: "Total Objects", value: "--", icon: Database },
  { label: "Average Confidence", value: "--", icon: ShieldCheck },
  { label: "Inference Time", value: "--", icon: Clock },
  { label: "GPU Memory", value: "--", icon: Monitor },
  { label: "Image Resolution", value: "--", icon: ImageIcon },
  { label: "Processing Date", value: "--", icon: Calendar },
]

const OVERALL_CONFIDENCE = 0
const CIRCUMFERENCE = 2 * Math.PI * 54 // r=54

/* ─────────────────────────────────────────────────────────────────────────
   Sub-components
───────────────────────────────────────────────────────────────────────── */

/** Animated SVG confidence ring */
const ConfidenceRing: React.FC<{ value: number }> = ({ value }) => {
  const offset = CIRCUMFERENCE - (value / 100) * CIRCUMFERENCE
  return (
    <div className="relative flex items-center justify-center w-36 h-36">
      <svg className="rotate-[-90deg] absolute" width="144" height="144">
        {/* Track */}
        <circle cx="72" cy="72" r="54" fill="none" stroke="rgba(34,211,238,0.08)" strokeWidth="8" />
        {/* Progress */}
        <motion.circle
          cx="72" cy="72" r="54" fill="none"
          stroke="url(#confGrad)"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          initial={{ strokeDashoffset: CIRCUMFERENCE }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.4, ease: "easeOut", delay: 0.3 }}
          style={{ filter: "drop-shadow(0 0 6px rgba(34,211,238,0.55))" }}
        />
        <defs>
          <linearGradient id="confGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#0EA5E9" />
            <stop offset="100%" stopColor="#14B8A6" />
          </linearGradient>
        </defs>
      </svg>
      <div className="text-center z-10">
        <motion.div
          className="text-2xl font-black text-white"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          {value > 0 ? `${value}%` : "--"}
        </motion.div>
        <div className="text-[10px] font-bold text-muted-foreground/80 uppercase tracking-wider mt-0.5">
          Confidence
        </div>
      </div>
    </div>
  )
}

/** Animated horizontal confidence bar */
const ConfBar: React.FC<{ value: number; color?: string; delay?: number }> = ({
  value,
  color = "from-cyan-500 to-teal-400",
  delay = 0,
}) => (
  <div className="flex items-center gap-2 w-full">
    <div className="flex-1 h-1.5 bg-navy-950 rounded-full overflow-hidden">
      <motion.div
        className={cn("h-full rounded-full bg-gradient-to-r", color)}
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 0.9, ease: "easeOut", delay }}
        style={{ boxShadow: "0 0 6px rgba(34,211,238,0.4)" }}
      />
    </div>
    <span className="text-[10px] font-bold text-muted-foreground/80 w-10 text-right shrink-0">
      {value}%
    </span>
  </div>
)

/* ─────────────────────────────────────────────────────────────────────────
   Page
───────────────────────────────────────────────────────────────────────── */
export const PredictionResult: React.FC = () => {
  const navigate = useNavigate()
  const [activeView, setActiveView] = useState<"original" | "annotated">("annotated")

  return (
    <div className="space-y-6">
      {/* Back button */}
      <div className="flex flex-wrap items-center justify-between gap-4 -mt-2">
        <Button
          variant="ghost" size="sm"
          onClick={() => navigate(-1)}
          className="text-muted-foreground hover:text-cyan-400 hover:bg-navy-800/40"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>
        <div className="px-3.5 py-1.5 rounded-xl border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-semibold flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 animate-pulse" />
          <span>Results will appear here after analysis.</span>
        </div>
      </div>

      {/* ── MAIN TWO-COLUMN GRID ── */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">

        {/* ═══════════════════════════════════════
            LEFT — Images (3/5)
        ════════════════════════════════════════ */}
        <div className="xl:col-span-3 space-y-5">

          {/* Toggle tabs */}
          <div className="flex items-center gap-1 p-1 bg-navy-950/50 border border-navy-800/80 rounded-xl w-fit">
            {(["original", "annotated"] as const).map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className={cn(
                  "px-4 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 capitalize",
                  activeView === view
                    ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/25 shadow-glow-sm/5"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {view === "original" ? "Original Image" : "Annotated Image"}
              </button>
            ))}
          </div>

          {/* Side-by-side comparison */}
          <Card className="overflow-hidden border-navy-800/80 p-0">
            <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-navy-800/50">
              {/* Original */}
              <div className="relative group">
                <div className="absolute top-3 left-3 z-10">
                  <span className="px-2 py-1 rounded-lg text-[10px] font-bold bg-navy-950/70 backdrop-blur-sm border border-navy-800/60 text-muted-foreground">
                    Original
                  </span>
                </div>
                <img
                  src={originalImg}
                  alt="Original uploaded image"
                  className="w-full h-64 object-cover object-center"
                />
              </div>

              {/* Annotated */}
              <div className="relative group">
                <div className="absolute top-3 left-3 z-10">
                  <span className="px-2 py-1 rounded-lg text-[10px] font-bold bg-cyan-500/20 backdrop-blur-sm border border-cyan-500/30 text-cyan-400">
                    AI Annotated
                  </span>
                </div>
                <img
                  src={annotatedImg}
                  alt="AI-annotated image with bounding boxes"
                  className="w-full h-64 object-cover object-center"
                />
                {/* Corner glow */}
                <div className="absolute inset-0 ring-1 ring-inset ring-cyan-500/20 pointer-events-none" />
              </div>
            </div>
            {/* Caption bar */}
            <div className="px-5 py-3 border-t border-navy-800/50 flex items-center justify-between text-xs text-muted-foreground/70">
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                Results will appear here after analysis.
              </span>
              <span className="font-mono">1920 × 1080</span>
            </div>
          </Card>

          {/* Detection Table */}
          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-base">
                <ScanSearch className="w-4 h-4 text-cyan-400" />
                Detection Results
              </CardTitle>
              <CardDescription>Per-object confidence scores from the ensemble pipeline.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-navy-800/80 text-[10px] uppercase tracking-wider text-muted-foreground/70">
                      <th className="text-left px-5 py-3 font-bold">Species</th>
                      <th className="text-left px-4 py-3 font-bold hidden sm:table-cell">Detector</th>
                      <th className="text-left px-4 py-3 font-bold hidden md:table-cell">Classifier</th>
                      <th className="text-left px-4 py-3 font-bold">Combined</th>
                      <th className="text-left px-4 py-3 font-bold hidden lg:table-cell">BBox ID</th>
                      <th className="text-center px-4 py-3 font-bold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-navy-800/30">
                    {DETECTIONS.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="py-8 text-center text-xs text-muted-foreground/60">
                          No data available
                        </td>
                      </tr>
                    ) : DETECTIONS.map((det, idx) => (
                      <motion.tr
                        key={det.id}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 + 0.2 }}
                        className={cn(
                          "group transition-all duration-200",
                          det.highlight
                            ? "bg-cyan-500/5 border-l-2 border-l-cyan-400"
                            : "hover:bg-navy-800/20"
                        )}
                      >
                        <td className="px-5 py-4">
                          <div className={cn("font-bold", det.highlight ? "text-cyan-400" : "text-foreground")}>
                            {det.species}
                          </div>
                          <div className="text-[10px] text-muted-foreground/60 italic mt-0.5">{det.scientific}</div>
                        </td>
                        <td className="px-4 py-4 hidden sm:table-cell">
                          <ConfBar value={det.detectorConf} delay={idx * 0.1 + 0.3} />
                        </td>
                        <td className="px-4 py-4 hidden md:table-cell">
                          <ConfBar
                            value={det.classifierConf}
                            color="from-teal-500 to-teal-300"
                            delay={idx * 0.1 + 0.4}
                          />
                        </td>
                        <td className="px-4 py-4">
                          <ConfBar
                            value={det.combinedConf}
                            color={det.highlight ? "from-cyan-400 to-teal-400" : "from-cyan-500/80 to-teal-500/80"}
                            delay={idx * 0.1 + 0.5}
                          />
                        </td>
                        <td className="px-4 py-4 hidden lg:table-cell">
                          <span className="font-mono text-muted-foreground/70">{det.id}</span>
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className={cn(
                            "inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-bold border",
                            det.status === "confirmed"
                              ? "bg-teal-500/10 text-teal-400 border-teal-500/20"
                              : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                          )}>
                            {det.status === "confirmed"
                              ? <CheckCircle2 className="w-2.5 h-2.5" />
                              : <Zap className="w-2.5 h-2.5" />}
                            <span className="capitalize text-[10px]">
                              {det.status === "confirmed" ? "Confirmed" : "Low Conf."}
                            </span>
                          </span>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ═══════════════════════════════════════
            RIGHT — Summary Panel (2/5)
        ════════════════════════════════════════ */}
        <div className="xl:col-span-2 space-y-5">

          {/* Confidence Summary */}
          <Card className="overflow-hidden">
            <CardContent className="pt-6 pb-5 flex flex-col items-center gap-4">
              {/* Processing badge */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ type: "spring", stiffness: 260, damping: 18 }}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/25 text-xs font-bold text-teal-400"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                Processing Complete
              </motion.div>

              {/* Ring */}
              <ConfidenceRing value={OVERALL_CONFIDENCE} />

              {/* Model badge */}
              <div className="w-full p-3 rounded-xl bg-navy-950/50 border border-navy-800/60 flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shrink-0">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">Model Version</div>
                  <div className="text-xs font-bold text-foreground">DeepSea Ensemble v1.0</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Pipeline Trust */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-teal-400" />
                Pipeline Trust Check
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 pb-5">
              {PIPELINE_CHECKS.map((check, idx) => (
                <motion.div
                  key={check.name}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 + 0.2 }}
                  className="flex items-center justify-between p-3 rounded-xl bg-navy-950/40 border border-navy-800/60 hover:border-teal-500/20 transition-colors"
                >
                  <div className="flex items-center gap-2.5">
                    <check.icon className="w-4 h-4 text-muted-foreground/60 shrink-0" />
                    <span className="text-xs font-semibold text-foreground">{check.name}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {/* Glowing green checkmark */}
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 300, damping: 15, delay: idx * 0.1 + 0.5 }}
                    >
                      <CheckCircle2
                        className="w-4 h-4 text-teal-400"
                        style={{ filter: "drop-shadow(0 0 5px rgba(45,212,191,0.7))" }}
                      />
                    </motion.div>
                    <span className="text-[10px] font-bold text-teal-400">{check.result}</span>
                  </div>
                </motion.div>
              ))}
            </CardContent>
          </Card>

          {/* Statistics */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Database className="w-4 h-4 text-cyan-400" />
                Analysis Statistics
              </CardTitle>
            </CardHeader>
            <CardContent className="pb-5">
              <div className="grid grid-cols-2 gap-2.5">
                {STATS.map((stat, idx) => (
                  <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.07 + 0.2 }}
                    className="p-3 rounded-xl bg-navy-950/50 border border-navy-800/60 space-y-1.5 hover:border-cyan-500/20 transition-colors"
                  >
                    <div className="flex items-center gap-1.5">
                      <stat.icon className="w-3.5 h-3.5 text-cyan-400/80 shrink-0" />
                      <span className="text-[10px] text-muted-foreground/70 truncate">{stat.label}</span>
                    </div>
                    <div className="text-xs font-bold text-foreground">{stat.value}</div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ── BOTTOM ACTION BUTTONS ── */}
      <div className="flex flex-wrap items-center gap-3 pt-2">
        <Button className="rounded-xl h-10 text-xs font-bold gap-2">
          <Download className="w-4 h-4" />
          Download Report
        </Button>
        <Button variant="secondary" className="rounded-xl h-10 text-xs font-semibold gap-2 border border-navy-700 bg-navy-800/60 hover:bg-navy-700 hover:text-cyan-400">
          <ImageIcon className="w-4 h-4" />
          Download Annotated Image
        </Button>
        <Button
          variant="outline"
          className="rounded-xl h-10 text-xs font-semibold gap-2 border-navy-700 hover:text-cyan-400 hover:border-cyan-500/30"
          onClick={() => navigate("/upload")}
        >
          <UploadCloud className="w-4 h-4" />
          Upload Another
        </Button>
        <Button variant="ghost" className="rounded-xl h-10 text-xs font-semibold gap-2 hover:text-cyan-400 hover:bg-navy-800/40">
          <Share2 className="w-4 h-4" />
          Share Result
        </Button>
      </div>
    </div>
  )
}
export default PredictionResult
