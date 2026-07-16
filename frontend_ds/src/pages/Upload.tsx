import React, { useState, useRef, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import {
  UploadCloud,
  ImageIcon,
  X,
  Sparkles,
  RefreshCw,
  FileImage,
  CheckCircle2
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { AILoadingScreen } from "@/components/ui/AILoadingScreen"
import sampleDetectionImg from "@/assets/sample_detection.png"

type UploadState = "empty" | "dragging" | "previewing" | "analyzing" | "done"

export const Upload: React.FC = () => {
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [uploadState, setUploadState] = useState<UploadState>("empty")
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>("")
  const [fileSize, setFileSize] = useState<string>("")

  /* ── File handling helpers ── */
  const processFile = useCallback((file: File) => {
    if (!file.type.match(/image\/(png|jpe?g)/i)) return
    setFileName(file.name)
    setFileSize((file.size / 1024).toFixed(1) + " KB")
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setUploadState("previewing")
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file) processFile(file)
      else setUploadState("empty")
    },
    [processFile]
  )

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }

  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setFileName("")
    setFileSize("")
    setUploadState("empty")
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleAnalyze = () => {
    setUploadState("analyzing")
  }

  const handleAnalysisComplete = () => {
    setUploadState("done")
  }

  /* ── Supported format badges ── */
  const formats = ["PNG", "JPG", "JPEG"]

  return (
    <div className="space-y-6 max-w-4xl mx-auto">

      {/* ── DRAG & DROP ZONE ── */}
      <AnimatePresence mode="wait">
        {(uploadState === "empty" || uploadState === "dragging") && (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3 }}
          >
            <div
              onDragOver={(e) => { e.preventDefault(); setUploadState("dragging") }}
              onDragLeave={() => setUploadState("empty")}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "relative flex flex-col items-center justify-center gap-5 p-14 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-300 overflow-hidden",
                uploadState === "dragging"
                  ? "border-cyan-400 bg-cyan-500/10 shadow-glow-md scale-[1.01]"
                  : "border-navy-700 hover:border-cyan-500/60 bg-navy-900/40 hover:bg-cyan-500/5 hover:shadow-glow-sm/10"
              )}
            >
              {/* Ambient glow blob behind icon */}
              <div className="absolute w-72 h-72 rounded-full bg-cyan-500/10 blur-[90px] pointer-events-none" />

              {/* Upload icon with ring */}
              <motion.div
                animate={uploadState === "dragging" ? { scale: 1.15 } : { scale: 1 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="relative flex items-center justify-center w-20 h-20 rounded-2xl bg-cyan-500/10 border border-cyan-500/25 text-cyan-400 shadow-glow-sm/5"
              >
                <UploadCloud className="w-9 h-9" strokeWidth={1.5} />
                {/* Animated ping ring on drag */}
                {uploadState === "dragging" && (
                  <span className="absolute inset-0 rounded-2xl border border-cyan-400/50 animate-ping" />
                )}
              </motion.div>

              {/* Text content */}
              <div className="text-center space-y-2">
                <p className="text-lg font-bold text-foreground">
                  {uploadState === "dragging"
                    ? "Release to Upload"
                    : "Drag & Drop Image Here"}
                </p>
                <p className="text-sm text-muted-foreground/80">
                  or click anywhere to browse your files
                </p>

                {/* Format badges */}
                <div className="flex items-center justify-center gap-2 pt-1">
                  <span className="text-[11px] text-muted-foreground/60">Supported:</span>
                  {formats.map((f) => (
                    <span
                      key={f}
                      className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>

              {/* Browse button */}
              <Button
                className="relative z-10 rounded-xl h-10 px-6 text-sm font-bold shadow-glow-sm"
                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
              >
                <FileImage className="w-4 h-4 mr-2" />
                Browse Files
              </Button>

              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>

            {/* ── EMPTY STATE: Sample Detection Preview ── */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.4 }}
              className="mt-6"
            >
              <Card className="overflow-hidden border-navy-800/80">
                <div className="relative">
                  <img
                    src={sampleDetectionImg}
                    alt="Sample AI detection output"
                    className="w-full object-cover max-h-72 object-center"
                  />
                  {/* Overlay gradient at bottom */}
                  <div className="absolute inset-0 bg-gradient-to-t from-navy-950/90 via-transparent to-transparent" />

                  {/* AI badge top-left */}
                  <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-navy-950/60 backdrop-blur-sm border border-cyan-500/25 text-xs font-bold text-cyan-400">
                    <Sparkles className="w-3 h-3" />
                    AI Detection Preview
                  </div>

                  {/* Caption bottom */}
                  <div className="absolute bottom-4 left-0 right-0 text-center px-6">
                    <p className="text-sm font-semibold text-white drop-shadow">
                      Upload an image to begin.
                    </p>
                    <p className="text-xs text-cyan-200/80 mt-0.5">
                      Ensemble detection identifies and labels marine species with bounding boxes
                    </p>
                  </div>
                </div>
              </Card>
            </motion.div>
          </motion.div>
        )}

        {/* ── PREVIEW STATE ── */}
        {(uploadState === "previewing" || uploadState === "analyzing" || uploadState === "done") && (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.35 }}
            className="space-y-5"
          >
            {/* Image Preview Card */}
            <Card className="overflow-hidden border-navy-800/80">
              <CardContent className="p-0">
                <div className="relative">
                  <img
                    src={previewUrl!}
                    alt="Uploaded preview"
                    className="w-full max-h-96 object-contain bg-navy-950/60"
                  />

                  {/* Gradient overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-navy-950/70 via-transparent to-transparent pointer-events-none" />

                  {/* Top bar: file info */}
                  <div className="absolute top-3 left-3 right-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-navy-950/70 backdrop-blur-sm border border-navy-800/60 text-xs">
                      <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
                      <span className="font-semibold text-foreground truncate max-w-[220px]">{fileName}</span>
                      <span className="text-muted-foreground/70">•</span>
                      <span className="text-muted-foreground/80">{fileSize}</span>
                    </div>

                    {/* Reset X button */}
                    <button
                      onClick={handleReset}
                      disabled={uploadState === "analyzing"}
                      className="w-8 h-8 rounded-xl bg-navy-950/70 backdrop-blur-sm border border-navy-800/60 flex items-center justify-center text-muted-foreground hover:text-rose-400 hover:border-rose-500/30 transition-all duration-200 disabled:opacity-40"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Enterprise AI loading screen */}
                  <AnimatePresence>
                    {uploadState === "analyzing" && (
                      <AILoadingScreen onComplete={handleAnalysisComplete} />
                    )}
                  </AnimatePresence>

                  {/* Done overlay */}
                  <AnimatePresence>
                    {uploadState === "done" && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 flex flex-col items-center justify-center bg-navy-950/75 backdrop-blur-sm"
                      >
                        <motion.div
                          initial={{ scale: 0.6, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ type: "spring", stiffness: 260, damping: 18 }}
                          className="w-16 h-16 rounded-full bg-teal-500/15 border border-teal-500/30 flex items-center justify-center shadow-glow-teal mb-4"
                        >
                          <CheckCircle2 className="w-8 h-8 text-teal-400" />
                        </motion.div>
                        <p className="text-sm font-bold text-white">Analysis Complete</p>
                        <p className="text-xs text-muted-foreground/80 mt-1">Prediction results are ready</p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex items-center gap-3">
              {uploadState === "done" ? (
                <Button
                  onClick={() => navigate("/prediction-result")}
                  className="flex-1 h-11 text-sm font-bold rounded-xl shadow-glow-sm"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  View Prediction Results
                </Button>
              ) : (
                <Button
                  onClick={handleAnalyze}
                  disabled={uploadState === "analyzing"}
                  className="flex-1 h-11 text-sm font-bold rounded-xl shadow-glow-sm disabled:opacity-60"
                >
                  {uploadState === "analyzing" ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 mr-2" />
                      Analyze Image
                    </>
                  )}
                </Button>
              )}

              <Button
                variant="outline"
                onClick={handleReset}
                disabled={uploadState === "analyzing"}
                className="h-11 px-5 text-sm font-semibold rounded-xl border-navy-700 hover:text-rose-400 hover:border-rose-500/30 disabled:opacity-40"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Reset
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  )
}
export default Upload
