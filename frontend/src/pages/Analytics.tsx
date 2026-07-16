import React, { useState } from "react"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area
} from "recharts"
import {
  Activity,
  TrendingUp,
  Clock,
  Award,
  Sparkles,
  PieChart as PieIcon,
  Flame,
  Calendar,
  ShieldCheck
} from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/* ─── Mock Datasets ─────────────────────────────────────────────────────── */

// 1. Uploads per Week & Avg Inference Time
const weeklyUploadsData: { week: string; uploads: number; avgLatency: number }[] = []

// 2. Species Distribution (Pie)
const speciesDistributionData: { name: string; value: number; color: string }[] = []

// 3. Confidence Distribution (Area Chart)
const confidenceDistData: { range: string; count: number }[] = []

// 4. Detection Accuracy across Model Iterations (Line)
const accuracyTrendData: { release: string; accuracy: number }[] = []

// 5. Timeline Activities
const TIMELINE_ACTIVITIES: { id: number; time: string; title: string; desc: string; type: string }[] = []

export const Analytics: React.FC = () => {
  const [timeRange, setTimeRange] = useState<"7d" | "30d" | "all">("7d")

  return (
    <div className="space-y-6">
      
      {/* ── HEADER ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-extrabold text-white tracking-tight">AI Analytics Dashboard</h2>
          <p className="text-xs text-muted-foreground/75 mt-0.5">
            Detailed performance charts, accuracy metrics, and detection distribution
          </p>
        </div>

        {/* Time Filter buttons */}
        <div className="flex items-center gap-1.5 p-1 bg-navy-950/50 border border-navy-800/80 rounded-xl self-start sm:self-auto">
          {(["7d", "30d", "all"] as const).map((r) => (
            <button
              key={r}
              onClick={() => setTimeRange(r)}
              className={cn(
                "px-3 py-1 rounded-lg text-xs font-bold transition-all duration-200 uppercase",
                timeRange === r
                  ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/25 shadow-glow-sm/5"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {r === "7d" ? "7 Days" : r === "30d" ? "30 Days" : "All Time"}
            </button>
          ))}
        </div>
      </div>

      {/* ── KPI HIGHLIGHTS CARDS ── */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Metric 1 */}
        <Card className="relative overflow-hidden group">
          <CardContent className="p-5 flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Avg Inference Time</span>
              <div className="text-2xl font-black text-white group-hover:text-cyan-400 transition-colors">--</div>
              <p className="text-[10px] text-muted-foreground/60 font-medium">No data available</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 shadow-glow-sm/5">
              <Clock className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>

        {/* Metric 2 */}
        <Card className="relative overflow-hidden group">
          <CardContent className="p-5 flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Overall Accuracy</span>
              <div className="text-2xl font-black text-white group-hover:text-teal-400 transition-colors">--</div>
              <p className="text-[10px] text-muted-foreground/60 font-medium">No data available</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 shadow-glow-sm/5">
              <Award className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>

        {/* Metric 3 */}
        <Card className="relative overflow-hidden group">
          <CardContent className="p-5 flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Mean Confidence</span>
              <div className="text-2xl font-black text-white group-hover:text-cyan-300 transition-colors">--</div>
              <p className="text-[10px] text-muted-foreground/60 font-medium">No data available</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-300 shadow-glow-sm/5">
              <ShieldCheck className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>

        {/* Metric 4 */}
        <Card className="relative overflow-hidden group">
          <CardContent className="p-5 flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Total Detections</span>
              <div className="text-2xl font-black text-white group-hover:text-indigo-400 transition-colors">0</div>
              <p className="text-[10px] text-muted-foreground/60 font-medium">No data available</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-glow-sm/5">
              <Sparkles className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── CHARTS SECTIONS GRID ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Chart 1: Uploads Per Week & Avg Inference Time */}
        <Card className="hover:border-navy-800 transition-colors">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="w-4 h-4 text-cyan-400" />
              Upload Trends & Inference Latency
            </CardTitle>
            <CardDescription>
              Correlation between weekly upload volume and average pipeline latency in milliseconds.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80 flex items-center justify-center">
            <div className="text-center space-y-2">
              <Activity className="w-8 h-8 text-muted-foreground/30 mx-auto" />
              <p className="text-sm font-semibold text-muted-foreground/50">No analytics available yet.</p>
              <p className="text-xs text-muted-foreground/35">Data will appear after images are uploaded and analysed.</p>
            </div>
          </CardContent>
        </Card>

        {/* Chart 2: Species Distribution (Pie) */}
        <Card className="hover:border-navy-800 transition-colors">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <PieIcon className="w-4 h-4 text-teal-400" />
              Detected Species Distribution
            </CardTitle>
            <CardDescription>
              Relative breakdown of marine life classifications verified across camera channels.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80 flex items-center justify-center">
            <div className="text-center space-y-2">
              <PieIcon className="w-8 h-8 text-muted-foreground/30 mx-auto" />
              <p className="text-sm font-semibold text-muted-foreground/50">No analytics available yet.</p>
              <p className="text-xs text-muted-foreground/35">Species distribution will populate once detections are made.</p>
            </div>
          </CardContent>
        </Card>

        {/* Chart 3: Confidence Distribution (Area) */}
        <Card className="hover:border-navy-800 transition-colors">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Flame className="w-4 h-4 text-cyan-400" />
              Confidence Distribution
            </CardTitle>
            <CardDescription>
              Volume count of classified entities falling into specific model prediction confidence boundaries.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80 flex items-center justify-center">
            <div className="text-center space-y-2">
              <Flame className="w-8 h-8 text-muted-foreground/30 mx-auto" />
              <p className="text-sm font-semibold text-muted-foreground/50">No analytics available yet.</p>
              <p className="text-xs text-muted-foreground/35">Confidence distribution will appear after model runs.</p>
            </div>
          </CardContent>
        </Card>

        {/* Chart 4: Detection Accuracy Trend (Bar/Line) */}
        <Card className="hover:border-navy-800 transition-colors">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="w-4 h-4 text-teal-400" />
              Pipeline Model Accuracy Evolution
            </CardTitle>
            <CardDescription>
              Validation set accuracy progression across different ensemble and classifier versions.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80 flex items-center justify-center">
            <div className="text-center space-y-2">
              <TrendingUp className="w-8 h-8 text-muted-foreground/30 mx-auto" />
              <p className="text-sm font-semibold text-muted-foreground/50">No analytics available yet.</p>
              <p className="text-xs text-muted-foreground/35">Model accuracy trends will populate once data is available.</p>
            </div>
          </CardContent>
        </Card>

      </div>

      {/* ── TIMELINE OF RECENT ACTIVITIES ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Calendar className="w-4 h-4 text-cyan-400" />
            Recent Pipeline Activity Timeline
          </CardTitle>
          <CardDescription>
            Audit log of deep-sea image classifications and system verification events.
          </CardDescription>
        </CardHeader>
        <CardContent className="pb-6">
          {TIMELINE_ACTIVITIES.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 space-y-2 text-center">
              <Calendar className="w-7 h-7 text-muted-foreground/30" />
              <p className="text-sm font-semibold text-muted-foreground/50">No analytics available.</p>
              <p className="text-xs text-muted-foreground/35">Activity will appear here after backend is connected.</p>
            </div>
          ) : (
            <div className="relative border-l border-navy-800/80 ml-3 pl-6 space-y-6">
              {TIMELINE_ACTIVITIES.map((act) => (
                <div key={act.id} className="relative group">
                  <div className="absolute -left-[33px] top-0.5 flex items-center justify-center">
                    <div className="w-5 h-5 rounded-full bg-[#071A2E] border border-navy-800 flex items-center justify-center group-hover:border-cyan-500/50 group-hover:shadow-glow-sm transition-all duration-300">
                      <div className={cn(
                        "w-2 h-2 rounded-full",
                        act.type === "success" ? "bg-teal-400" :
                        act.type === "info" ? "bg-cyan-400" :
                        act.type === "system" ? "bg-indigo-400" : "bg-amber-400"
                      )} />
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                      <h4 className="text-xs font-bold text-foreground group-hover:text-cyan-400 transition-colors">
                        {act.title}
                      </h4>
                      <span className="text-[10px] text-muted-foreground/60 font-medium">
                        {act.time}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground/80 leading-relaxed max-w-2xl">
                      {act.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
      
    </div>
  )
}
export default Analytics
