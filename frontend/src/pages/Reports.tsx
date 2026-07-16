import React, { useState } from "react"
import {
  FileText,
  FileSpreadsheet,
  Image as ImageIcon,
  Database,
  Download,
  RefreshCw,
  Calendar,
  CheckCircle2,
  Clock,
  FileArchive,
  ShieldCheck
} from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/* ─── Mock Data ─────────────────────────────────────────────────────────── */
const DOWNLOADS_HISTORY: { name: string; type: string; size: string; date: string; status: string }[] = []

export const Reports: React.FC = () => {
  const [compilingCard, setCompilingCard] = useState<string | null>(null)

  const handleCompile = (cardName: string) => {
    setCompilingCard(cardName)
    setTimeout(() => {
      setCompilingCard(null)
    }, 2000)
  }

  return (
    <div className="space-y-6">
      
      {/* ── HEADER ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-extrabold text-white tracking-tight">Reports & Exports Center</h2>
          <p className="text-xs text-muted-foreground/75 mt-0.5">
            Download model logs, export tabular results, compile annotated media packs, or package training datasets.
          </p>
        </div>

        {/* Quick status indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/20 text-xs font-bold text-teal-400">
          <ShieldCheck className="w-4 h-4" />
          Download Queue Idle
        </div>
      </div>

      {/* ── METRIC SNAPSHOTS ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 flex flex-col justify-between hover:border-cyan-500/20 transition-all duration-300">
          <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Total Downloads Ready</span>
          <div className="mt-2 text-2xl font-black text-white">0</div>
          <p className="text-[10px] text-muted-foreground/60 mt-1">No reports generated yet</p>
        </div>
        <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 flex flex-col justify-between hover:border-cyan-500/20 transition-all duration-300">
          <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Storage Utilized</span>
          <div className="mt-2 text-2xl font-black text-white">--</div>
          <p className="text-[10px] text-muted-foreground/60 mt-1">No data available</p>
        </div>
        <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 flex flex-col justify-between hover:border-cyan-500/20 transition-all duration-300">
          <span className="text-[10px] font-bold text-muted-foreground/70 uppercase tracking-wider">Last Generated</span>
          <div className="mt-2 text-2xl font-black text-white">--</div>
          <p className="text-[10px] text-muted-foreground/60 mt-1">No report generated yet</p>
        </div>
      </div>

      {/* ── PRIMARY EXPORT SECTIONS ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* 1. PDF Reports */}
        <Card className="hover:border-navy-800 transition-all duration-300 flex flex-col justify-between">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 shadow-glow-sm/5">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <CardTitle className="text-base">PDF Analysis Reports</CardTitle>
                <CardDescription>Formatted executive summary documents.</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col justify-between">
            <p className="text-xs text-muted-foreground/80 leading-relaxed">
              Generates high-fidelity PDF documents featuring summary charts, species distribution matrixes, model confidence statistics, and chronological logs.
            </p>
            <div className="space-y-3 pt-2">
              <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider">Latest PDF</div>
              <div className="flex items-center justify-between p-3 rounded-xl bg-navy-950/40 border border-navy-800/60 text-xs">
                <div>
                  <div className="font-bold text-muted-foreground/60">No report generated yet</div>
                  <div className="text-[10px] text-muted-foreground/40 mt-0.5">--</div>
                </div>
                <Button size="sm" disabled className="h-8 rounded-lg text-xs font-semibold gap-1.5 opacity-40" aria-disabled="true">
                  <Download className="w-3.5 h-3.5" />
                  Download
                </Button>
              </div>

              <Button
                variant="outline"
                onClick={() => handleCompile("pdf")}
                disabled={compilingCard === "pdf"}
                className="w-full h-9 text-xs rounded-xl font-bold border-navy-700 hover:text-cyan-400"
              >
                <RefreshCw className={cn("w-3.5 h-3.5 mr-2", compilingCard === "pdf" && "animate-spin")} />
                {compilingCard === "pdf" ? "Rebuilding PDF..." : "Compile Fresh Q2 Report"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 2. CSV Tabular Results */}
        <Card className="hover:border-navy-800 transition-all duration-300 flex flex-col justify-between">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 shadow-glow-sm/5">
                <FileSpreadsheet className="w-5 h-5" />
              </div>
              <div>
                <CardTitle className="text-base">CSV Tabular Results</CardTitle>
                <CardDescription>Raw tabular database outputs.</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col justify-between">
            <p className="text-xs text-muted-foreground/80 leading-relaxed">
              Export bounding box coordinates, model labels, individual confidence scores, and precise detection timestamps in standard comma-separated sheets.
            </p>
            <div className="space-y-3 pt-2">
              <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider">Latest CSV</div>
              <div className="flex items-center justify-between p-3 rounded-xl bg-navy-950/40 border border-navy-800/60 text-xs">
                <div>
                  <div className="font-bold text-muted-foreground/60">No report generated yet</div>
                  <div className="text-[10px] text-muted-foreground/40 mt-0.5">--</div>
                </div>
                <Button size="sm" disabled className="h-8 rounded-lg text-xs font-semibold gap-1.5 opacity-40" aria-disabled="true">
                  <Download className="w-3.5 h-3.5" />
                  Download
                </Button>
              </div>

              <Button
                variant="outline"
                onClick={() => handleCompile("csv")}
                disabled={compilingCard === "csv"}
                className="w-full h-9 text-xs rounded-xl font-bold border-navy-700 hover:text-teal-400"
              >
                <RefreshCw className={cn("w-3.5 h-3.5 mr-2", compilingCard === "csv" && "animate-spin")} />
                {compilingCard === "csv" ? "Re-exporting CSV..." : "Export Current Table logs"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 3. Annotated Images */}
        <Card className="hover:border-navy-800 transition-all duration-300 flex flex-col justify-between">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 shadow-glow-sm/5">
                <ImageIcon className="w-5 h-5" />
              </div>
              <div>
                <CardTitle className="text-base">Annotated Images ZIP</CardTitle>
                <CardDescription>Media archives with visual annotations.</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col justify-between">
            <p className="text-xs text-muted-foreground/80 leading-relaxed">
              Batch compress and download verified marine species imagery with labeled YOLO model bounding boxes burned directly into the output frames.
            </p>
            <div className="space-y-3 pt-2">
              <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider">Latest ZIP Package</div>
              <div className="flex items-center justify-between p-3 rounded-xl bg-navy-950/40 border border-navy-800/60 text-xs">
                <div>
                  <div className="font-bold text-muted-foreground/60">No archive generated yet</div>
                  <div className="text-[10px] text-muted-foreground/40 mt-0.5">--</div>
                </div>
                <Button size="sm" disabled className="h-8 rounded-lg text-xs font-semibold gap-1.5 opacity-40" aria-disabled="true">
                  <Download className="w-3.5 h-3.5" />
                  Download
                </Button>
              </div>

              <Button
                variant="outline"
                onClick={() => handleCompile("media")}
                disabled={compilingCard === "media"}
                className="w-full h-9 text-xs rounded-xl font-bold border-navy-700 hover:text-cyan-400"
              >
                <RefreshCw className={cn("w-3.5 h-3.5 mr-2", compilingCard === "media" && "animate-spin")} />
                {compilingCard === "media" ? "Archiving Media..." : "Compress & Zip Frame Cache"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 4. Dataset Export */}
        <Card className="hover:border-navy-800 transition-all duration-300 flex flex-col justify-between">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-glow-sm/5">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <CardTitle className="text-base">ML Dataset Exports</CardTitle>
                <CardDescription>COCO & YOLO format standard layouts.</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col justify-between">
            <p className="text-xs text-muted-foreground/80 leading-relaxed">
              Export subsea annotations to train target models. Includes images mapped alongside text/json metadata folders formatted for YOLOv8 or COCO.
            </p>
            <div className="space-y-3 pt-2">
              <div className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider">Active Export Structure</div>
              <div className="flex items-center justify-between p-3 rounded-xl bg-navy-950/40 border border-navy-800/60 text-xs">
                <div>
                  <div className="font-bold text-muted-foreground/60">No dataset exported yet</div>
                  <div className="text-[10px] text-muted-foreground/40 mt-0.5">--</div>
                </div>
                <Button size="sm" disabled className="h-8 rounded-lg text-xs font-semibold gap-1.5 opacity-40" aria-disabled="true">
                  <FileArchive className="w-3.5 h-3.5" />
                  Export ZIP
                </Button>
              </div>

              <Button
                variant="outline"
                onClick={() => handleCompile("dataset")}
                disabled={compilingCard === "dataset"}
                className="w-full h-9 text-xs rounded-xl font-bold border-navy-700 hover:text-indigo-400"
              >
                <RefreshCw className={cn("w-3.5 h-3.5 mr-2", compilingCard === "dataset" && "animate-spin")} />
                {compilingCard === "dataset" ? "Packaging Dataset..." : "Package Complete Image Repository"}
              </Button>
            </div>
          </CardContent>
        </Card>

      </div>

      {/* ── DOWNLOADS HISTORY & ACTIVITY LOG ── */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-sm flex items-center gap-2">
            <Calendar className="w-4 h-4 text-cyan-400" />
            File Generation & Download History
          </CardTitle>
          <CardDescription>
            History of generated reports and dataset compilation archives in this console.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-navy-800/80 text-[10px] uppercase tracking-wider text-muted-foreground/70">
                  <th className="text-left px-5 py-3 font-bold">File Name</th>
                  <th className="text-left px-4 py-3 font-bold">Type</th>
                  <th className="text-left px-4 py-3 font-bold">Size</th>
                  <th className="text-left px-4 py-3 font-bold">Generated</th>
                  <th className="text-center px-4 py-3 font-bold">Status</th>
                  <th className="text-center px-5 py-3 font-bold">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-800/30">
                {DOWNLOADS_HISTORY.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-xs text-muted-foreground/50">
                      No reports generated yet.
                    </td>
                  </tr>
                ) : DOWNLOADS_HISTORY.map((hist, idx) => (
                  <tr key={idx} className="hover:bg-navy-800/10 transition-colors">
                    <td className="px-5 py-3.5 font-bold text-foreground truncate max-w-[200px]">
                      {hist.name}
                    </td>
                    <td className="px-4 py-3.5 text-muted-foreground/80">{hist.type}</td>
                    <td className="px-4 py-3.5 font-mono text-muted-foreground/70">{hist.size}</td>
                    <td className="px-4 py-3.5 text-muted-foreground/60">{hist.date}</td>
                    <td className="px-4 py-3.5 text-center">
                      <span className={cn(
                        "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-bold border text-[10px]",
                        hist.status === "available"
                          ? "bg-teal-500/10 text-teal-400 border-teal-500/20"
                          : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                      )}>
                        {hist.status === "available" ? (
                          <><CheckCircle2 className="w-2.5 h-2.5" />Available</>
                        ) : (
                          <><Clock className="w-2.5 h-2.5" />Expired</>
                        )}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-center">
                      <Button
                        size="sm"
                        disabled={hist.status === "expired"}
                        className="h-7 px-2.5 rounded-lg text-[10px] font-bold gap-1 shadow-glow-sm disabled:opacity-40"
                      >
                        <Download className="w-3 h-3" />
                        Download
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
export default Reports
