import React from "react"
import { useNavigate } from "react-router-dom"
import {
  UploadCloud,
  History,
  BarChart3,
  FileText,
  ArrowRight,
  Cpu,
  Layers,
  Sparkles,
  TrendingUp,
  Image as ImageIcon,
  Clock,
  Eye,
  CheckCircle2
} from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export const Dashboard: React.FC = () => {
  const navigate = useNavigate()

  const overviewMetrics = [
    {
      title: "Total Images Uploaded",
      value: "0",
      change: "No data available",
      icon: ImageIcon,
      color: "text-cyan-400"
    },
    {
      title: "Total Species Detected",
      value: "0",
      change: "No data available",
      icon: Sparkles,
      color: "text-teal-400"
    },
    {
      title: "Average Confidence",
      value: "--",
      change: "No data available",
      icon: TrendingUp,
      color: "text-cyan-300"
    },
    {
      title: "Average Inference Time",
      value: "--",
      change: "No data available",
      icon: Clock,
      color: "text-teal-300"
    }
  ]

  const recentPredictions: {
    id: string
    species: string
    confidence: string
    date: string
    status: string
    thumbGradient: string
  }[] = []

  const quickActions = [
    {
      title: "Upload Image",
      description: "Analyze individual files or batch captures",
      path: "/upload",
      icon: UploadCloud,
      color: "from-cyan-500/20 to-cyan-500/5 hover:border-cyan-500/40 text-cyan-400"
    },
    {
      title: "Classification History",
      description: "Browse log archives and export datasets",
      path: "/history",
      icon: History,
      color: "from-teal-500/20 to-teal-500/5 hover:border-teal-500/40 text-teal-400"
    },
    {
      title: "Analytics Graphs",
      description: "Review model metrics and population dynamics",
      path: "/analytics",
      icon: BarChart3,
      color: "from-cyan-500/20 to-cyan-500/5 hover:border-cyan-500/40 text-cyan-300"
    },
    {
      title: "Compiled Reports",
      description: "Download PDFs and population matrices",
      path: "/reports",
      icon: FileText,
      color: "from-teal-500/20 to-teal-500/5 hover:border-teal-500/40 text-teal-300"
    }
  ]

  const pipelineSteps = [
    { id: "1", name: "YOLOv8-n", label: "Object Detector (Nano)", icon: Cpu },
    { id: "2", name: "YOLOv8-s", label: "Object Detector (Small)", icon: Cpu },
    { id: "3", name: "Weighted Box Fusion", label: "Ensemble Box Merger", icon: Layers },
    { id: "4", name: "Species Classifier", label: "CNN Categorizer", icon: Sparkles },
    { id: "5", name: "Final Prediction", label: "Bounding Box & Label", icon: CheckCircle2 }
  ]

  return (
    <div className="space-y-6">
      
      {/* ── TOP GREETING ── */}
      <div className="relative p-6 rounded-2xl border border-cyan-500/25 bg-navy-900/40 backdrop-blur-xl shadow-glow-sm/5 overflow-hidden flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        {/* Glow dots inside greeting */}
        <div className="absolute right-0 top-0 w-64 h-64 rounded-full bg-cyan-500/10 blur-[80px] pointer-events-none" />
        <div className="space-y-1 relative z-10">
          <h2 className="text-2xl font-extrabold tracking-tight text-white">
            Welcome Back, Operator
          </h2>
          <p className="text-xs text-muted-foreground/80">
            System connected to Pacific Northwest subsea imaging arrays. All pipelines running nominal.
          </p>
        </div>
        <Button
          onClick={() => navigate("/upload")}
          className="relative z-10 rounded-xl h-10 px-5 text-xs font-bold"
        >
          Analyze New Capture
          <ArrowRight className="w-4 h-4 ml-1.5" />
        </Button>
      </div>

      {/* ── OVERVIEW CARDS ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {overviewMetrics.map((m, idx) => (
          <Card key={idx} className="hover:border-cyan-500/30 transition-all duration-300">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/90">
                {m.title}
              </span>
              <m.icon className={`w-5 h-5 ${m.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-black tracking-tight text-foreground">{m.value}</div>
              <p className="text-xs text-muted-foreground/80 mt-1">{m.change}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* ── QUICK ACTIONS ── */}
      <div className="space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground/90 px-1">
          Quick Actions
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickActions.map((a, idx) => (
            <div
              key={idx}
              onClick={() => navigate(a.path)}
              className={`p-4 rounded-2xl bg-gradient-to-tr ${a.color} border border-navy-800/80 cursor-pointer shadow-card transition-all duration-300 flex items-start gap-4 hover:-translate-y-0.5 hover:shadow-glow-sm/5`}
            >
              <div className="w-9 h-9 rounded-xl bg-navy-950/60 border border-navy-800/60 flex items-center justify-center shrink-0">
                <a.icon className="w-5 h-5" />
              </div>
              <div className="space-y-0.5">
                <h4 className="text-xs font-bold text-foreground">{a.title}</h4>
                <p className="text-[11px] text-muted-foreground/80 leading-relaxed">{a.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── RECENT PREDICTIONS TABLE ── */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Recent Image Classifications</CardTitle>
              <CardDescription>Review the latest computer vision outputs from subsea camera captures.</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate("/history")}
              className="text-xs border-navy-800 rounded-lg hover:text-cyan-400"
            >
              View All History
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="border-b border-navy-800/80 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/90">
                  <th className="py-3 px-4 w-16">Thumbnail</th>
                  <th className="py-3 px-4">Species Detection Label</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4 text-center">Status</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-800/40">
                {recentPredictions.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-xs text-muted-foreground/60">
                      No data available
                    </td>
                  </tr>
                ) : (
                  recentPredictions.map((pred) => (
                    <tr key={pred.id} className="hover:bg-navy-800/20 hover:text-cyan-400 transition-colors group">
                      <td className="py-3 px-4">
                        {/* Premium mock thumbnail */}
                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-tr ${pred.thumbGradient} border border-navy-800/60 flex items-center justify-center shadow-inner`}>
                          <ImageIcon className="w-4.5 h-4.5" />
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <div className="font-semibold text-foreground group-hover:text-cyan-450">{pred.species}</div>
                        <div className="text-[10px] text-muted-foreground/80 font-mono mt-0.5">{pred.id}</div>
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-foreground/90 group-hover:text-cyan-400">
                        {pred.confidence}
                      </td>
                      <td className="py-3 px-4 text-xs text-muted-foreground">{pred.date}</td>
                      <td className="py-3 px-4 text-center">
                        <span className={cn(
                          "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border",
                          pred.status === "completed"
                            ? "bg-teal-500/10 text-teal-400 border-teal-500/20"
                            : "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                        )}>
                          {pred.status === "completed" ? (
                            <CheckCircle2 className="w-3 h-3" />
                          ) : (
                            <Clock className="w-3 h-3 animate-spin" />
                          )}
                          <span className="capitalize">{pred.status}</span>
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        {pred.status === "completed" ? (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => navigate("/prediction-result")}
                            className="h-8 text-xs hover:bg-cyan-500/10 hover:text-cyan-400 rounded-lg gap-1.5"
                          >
                            <Eye className="w-3.5 h-3.5" />
                            View Result
                          </Button>
                        ) : (
                          <Button variant="ghost" size="sm" disabled className="h-8 text-xs rounded-lg">
                            Processing
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* ── BOTTOM SECTION: MODEL PIPELINE ── */}
      <Card>
        <CardHeader>
          <CardTitle>Inference Model Pipeline</CardTitle>
          <CardDescription>
            DeepSea image classification employs an ensemble bounding box merger (Weighted Box Fusion) combined with CNN classifications.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col xl:flex-row items-center justify-between gap-4 p-6 rounded-2xl bg-navy-950/40 border border-navy-800/80 relative overflow-hidden">
            {/* Ambient pipeline background glow */}
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-teal-500/5 to-transparent pointer-events-none" />
            
            {pipelineSteps.map((step, idx) => (
              <React.Fragment key={step.id}>
                {/* Pipeline Step Card */}
                <div className="flex items-center gap-3 p-4 rounded-xl bg-navy-900 border border-navy-800/80 shadow-md relative z-10 w-full xl:w-56 hover:border-cyan-500/25 transition-all duration-300">
                  <div className="w-8 h-8 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 flex items-center justify-center shrink-0">
                    <step.icon className="w-4 h-4" />
                  </div>
                  <div className="text-left overflow-hidden">
                    <div className="text-xs font-bold text-foreground truncate">{step.name}</div>
                    <div className="text-[10px] text-muted-foreground/80 truncate mt-0.5">{step.label}</div>
                  </div>
                </div>

                {/* Glowing arrow connector - Hidden on final item */}
                {idx < pipelineSteps.length - 1 && (
                  <div className="flex items-center justify-center shrink-0 relative py-1 xl:py-0">
                    {/* Horizontal Arrow on wide screen, vertical arrow on small screen */}
                    <ArrowRight className="w-5 h-5 text-cyan-400 drop-shadow-[0_0_6px_#22d3ee] hidden xl:block animate-pulse" />
                    <div className="w-0.5 h-6 bg-cyan-400/30 shadow-[0_0_8px_#22d3ee] xl:hidden block animate-pulse" />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
export default Dashboard
