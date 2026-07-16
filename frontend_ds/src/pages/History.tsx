import React, { useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import {
  Search,
  LayoutGrid,
  List,
  Eye,
  Download,
  Trash2,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Filter,
  X,
  Image as ImageIcon,
  ChevronDown,
  ScanSearch
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import underwaterImg from "@/assets/underwater.png"
import sampleDetectionImg from "@/assets/sample_detection.png"

/* ─── Types ─────────────────────────────────────────────────────────────── */
type Status = "completed" | "analyzing" | "failed"
type ViewMode = "grid" | "list"

interface HistoryEntry {
  id: string
  filename: string
  species: string
  detectionCount: number
  confidence: number
  date: string
  status: Status
  thumb: string
}

/* ─── Mock Data ─────────────────────────────────────────────────────────── */
const MOCK_DATA: HistoryEntry[] = []

const ALL_SPECIES = ["All Species", ...Array.from(new Set(MOCK_DATA.map(d => d.species).filter(s => s !== "Analyzing..." && s !== "Unknown")))]
const ALL_STATUSES = ["All Status", "completed", "analyzing", "failed"]
const DATE_RANGES  = ["All Time", "Today", "Last 7 Days", "Last 30 Days"]

/* ─── Helpers ───────────────────────────────────────────────────────────── */
const statusMeta: Record<Status, { label: string; className: string; icon: React.ElementType }> = {
  completed: { label: "Completed", className: "bg-teal-500/10 text-teal-400 border-teal-500/20",   icon: CheckCircle2 },
  analyzing: { label: "Analyzing", className: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",   icon: Clock },
  failed:    { label: "Failed",    className: "bg-rose-500/10  text-rose-400  border-rose-500/20",  icon: AlertTriangle },
}

const ConfBar: React.FC<{ value: number }> = ({ value }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-1 bg-navy-950 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-teal-400 transition-all duration-700"
        style={{ width: `${value}%`, boxShadow: "0 0 5px rgba(34,211,238,0.4)" }}
      />
    </div>
    <span className="text-[10px] font-bold text-foreground w-9 text-right">{value > 0 ? `${value}%` : "—"}</span>
  </div>
)

/* ─── Dropdown ──────────────────────────────────────────────────────────── */
const FilterSelect: React.FC<{ value: string; options: string[]; onChange: (v: string) => void }> = ({
  value, options, onChange
}) => (
  <div className="relative">
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="appearance-none h-9 pl-3 pr-8 text-xs font-semibold bg-navy-900/50 border border-navy-800 rounded-xl text-foreground focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10 outline-none cursor-pointer transition-all duration-200"
    >
      {options.map(o => <option key={o} value={o} className="bg-navy-900">{o}</option>)}
    </select>
    <ChevronDown className="absolute right-2.5 top-2.5 w-3.5 h-3.5 text-muted-foreground/60 pointer-events-none" />
  </div>
)

/* ─── Grid Card ─────────────────────────────────────────────────────────── */
const GridCard: React.FC<{ entry: HistoryEntry; onView: () => void; onDelete: () => void }> = ({ entry, onView, onDelete }) => {
  const meta = statusMeta[entry.status]
  return (
    <Card className="overflow-hidden border-navy-800/80 hover:border-cyan-500/25 hover:shadow-card-hover transition-all duration-300 group cursor-pointer" onClick={onView}>
      {/* Thumbnail */}
      <div className="relative overflow-hidden h-44">
        <img src={entry.thumb} alt={entry.species} className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-500" />
        <div className="absolute inset-0 bg-gradient-to-t from-navy-950/90 via-navy-950/20 to-transparent" />
        {/* Status badge top-right */}
        <span className={cn("absolute top-2.5 right-2.5 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border backdrop-blur-sm", meta.className)}>
          <meta.icon className="w-2.5 h-2.5" />
          {meta.label}
        </span>
        {/* ID badge top-left */}
        <span className="absolute top-2.5 left-2.5 px-2 py-0.5 rounded-lg text-[10px] font-mono font-bold bg-navy-950/70 backdrop-blur-sm border border-navy-800/60 text-muted-foreground">
          {entry.id}
        </span>
        {/* Detection count overlay bottom */}
        {entry.status === "completed" && (
          <div className="absolute bottom-2.5 left-2.5 flex items-center gap-1 px-2 py-0.5 rounded-lg bg-navy-950/70 backdrop-blur-sm border border-cyan-500/20 text-[10px] font-bold text-cyan-400">
            <ScanSearch className="w-2.5 h-2.5" />
            {entry.detectionCount} detected
          </div>
        )}
      </div>

      <CardContent className="p-4 space-y-3">
        <div>
          <h4 className="text-sm font-bold text-foreground group-hover:text-cyan-400 transition-colors truncate">
            {entry.species}
          </h4>
          <p className="text-[10px] text-muted-foreground/70 truncate mt-0.5">{entry.filename}</p>
        </div>

        <ConfBar value={entry.confidence} />

        <p className="text-[10px] text-muted-foreground/60">{entry.date}</p>

        {/* Action buttons */}
        <div className="flex items-center gap-1.5 pt-1" onClick={e => e.stopPropagation()}>
          <Button size="sm" onClick={onView} className="flex-1 h-7 text-[10px] font-bold rounded-lg gap-1">
            <Eye className="w-3 h-3" />
            View
          </Button>
          <Button variant="outline" size="sm" className="h-7 w-7 p-0 rounded-lg border-navy-700 hover:text-cyan-400 hover:border-cyan-500/30">
            <Download className="w-3 h-3" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onDelete} className="h-7 w-7 p-0 rounded-lg hover:text-rose-400 hover:bg-rose-500/10">
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

/* ─── List Row ──────────────────────────────────────────────────────────── */
const ListRow: React.FC<{ entry: HistoryEntry; onView: () => void; onDelete: () => void }> = ({ entry, onView, onDelete }) => {
  const meta = statusMeta[entry.status]
  return (
    <div
      onClick={onView}
      className="flex items-center gap-4 p-4 rounded-2xl border border-navy-800/80 bg-navy-900/40 hover:border-cyan-500/25 hover:bg-navy-900/60 hover:shadow-card-hover transition-all duration-200 cursor-pointer group"
    >
      {/* Thumb */}
      <div className="w-16 h-16 rounded-xl overflow-hidden shrink-0 border border-navy-800/60">
        <img src={entry.thumb} alt={entry.species} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
      </div>

      {/* Main info */}
      <div className="flex-1 min-w-0 space-y-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-bold text-foreground group-hover:text-cyan-400 transition-colors truncate">{entry.species}</span>
          <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border shrink-0", meta.className)}>
            <meta.icon className="w-2.5 h-2.5" />
            {meta.label}
          </span>
        </div>
        <p className="text-[10px] text-muted-foreground/70 truncate">{entry.filename}</p>
        <p className="text-[10px] text-muted-foreground/60">{entry.date}</p>
      </div>

      {/* Stats */}
      <div className="hidden sm:flex flex-col items-end gap-1 shrink-0 w-32">
        <div className="flex items-center gap-1 text-[10px] text-muted-foreground/70">
          <ScanSearch className="w-3 h-3 text-cyan-400" />
          {entry.detectionCount > 0 ? `${entry.detectionCount} objects` : "—"}
        </div>
        <ConfBar value={entry.confidence} />
      </div>

      {/* ID */}
      <span className="hidden lg:block font-mono text-[10px] text-muted-foreground/50 shrink-0 w-20 text-right">{entry.id}</span>

      {/* Actions */}
      <div className="flex items-center gap-1.5 shrink-0" onClick={e => e.stopPropagation()}>
        <Button size="sm" onClick={onView} className="h-8 text-[10px] font-bold rounded-lg gap-1.5 px-3">
          <Eye className="w-3 h-3" />
          <span className="hidden sm:inline">View</span>
        </Button>
        <Button variant="outline" size="sm" className="h-8 w-8 p-0 rounded-lg border-navy-700 hover:text-cyan-400">
          <Download className="w-3 h-3" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onDelete} className="h-8 w-8 p-0 rounded-lg hover:text-rose-400 hover:bg-rose-500/10">
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>
    </div>
  )
}

/* ─── Page ───────────────────────────────────────────────────────────────── */
export const History: React.FC = () => {
  const navigate = useNavigate()

  const [view, setView]             = useState<ViewMode>("grid")
  const [search, setSearch]         = useState("")
  const [speciesFilter, setSpecies] = useState("All Species")
  const [statusFilter, setStatus]   = useState("All Status")
  const [dateFilter, setDate]       = useState("All Time")
  const [items, setItems]           = useState(MOCK_DATA)

  const filtered = useMemo(() => {
    return items.filter(d => {
      const matchSearch  = search === "" || d.species.toLowerCase().includes(search.toLowerCase()) || d.filename.toLowerCase().includes(search.toLowerCase()) || d.id.toLowerCase().includes(search.toLowerCase())
      const matchSpecies = speciesFilter === "All Species" || d.species === speciesFilter
      const matchStatus  = statusFilter  === "All Status"  || d.status  === statusFilter
      return matchSearch && matchSpecies && matchStatus
    })
  }, [items, search, speciesFilter, statusFilter, dateFilter])

  const handleDelete = (id: string) => setItems(prev => prev.filter(d => d.id !== id))
  const handleView   = () => navigate("/prediction-result")
  const clearFilters = () => { setSearch(""); setSpecies("All Species"); setStatus("All Status"); setDate("All Time") }
  const hasFilters   = search !== "" || speciesFilter !== "All Species" || statusFilter !== "All Status" || dateFilter !== "All Time"

  return (
    <div className="space-y-5">

      {/* ── HEADER ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-extrabold text-foreground tracking-tight">Classification History</h2>
          <p className="text-xs text-muted-foreground/75 mt-0.5">
            {filtered.length} of {items.length} captures
          </p>
        </div>

        {/* View toggle */}
        <div className="flex items-center gap-1.5 p-1 bg-navy-950/50 border border-navy-800/80 rounded-xl self-start sm:self-auto">
          <button
            onClick={() => setView("grid")}
            className={cn("h-7 w-7 flex items-center justify-center rounded-lg transition-all duration-200",
              view === "grid" ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/25" : "text-muted-foreground hover:text-foreground")}
          >
            <LayoutGrid className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setView("list")}
            className={cn("h-7 w-7 flex items-center justify-center rounded-lg transition-all duration-200",
              view === "list" ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/25" : "text-muted-foreground hover:text-foreground")}
          >
            <List className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── SEARCH + FILTERS ── */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 flex-wrap">
        {/* Search */}
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-muted-foreground/70" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search species, file name or ID..."
            className="w-full h-9 pl-9 pr-4 text-xs bg-navy-900/50 border border-navy-800 rounded-xl text-foreground placeholder:text-muted-foreground/60 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10 outline-none transition-all duration-200"
          />
        </div>

        {/* Filters row */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider">
            <Filter className="w-3 h-3" />
          </div>
          <FilterSelect value={speciesFilter} options={ALL_SPECIES}   onChange={setSpecies} />
          <FilterSelect value={dateFilter}    options={DATE_RANGES}   onChange={setDate}    />
          <FilterSelect value={statusFilter}  options={ALL_STATUSES}  onChange={setStatus}  />
          {hasFilters && (
            <button onClick={clearFilters} className="h-9 px-3 flex items-center gap-1.5 text-[10px] font-bold text-rose-400 hover:bg-rose-500/10 rounded-xl border border-rose-500/20 transition-all duration-200">
              <X className="w-3 h-3" />
              Clear
            </button>
          )}
        </div>
      </div>

      {/* ── GALLERY ── */}
      <AnimatePresence mode="wait">
        {items.length === 0 ? (
          <motion.div
            key="empty-upload"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-24 space-y-4 text-center"
          >
            <div className="w-14 h-14 rounded-2xl bg-navy-900/60 border border-navy-800/80 flex items-center justify-center">
              <ImageIcon className="w-6 h-6 text-muted-foreground/40" />
            </div>
            <p className="text-sm font-semibold text-muted-foreground/60">No images uploaded yet.</p>
            <p className="text-xs text-muted-foreground/40">Upload an image in the upload portal to begin AI analysis.</p>
          </motion.div>
        ) : filtered.length === 0 ? (
          <motion.div
            key="empty-filters"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-24 space-y-4 text-center"
          >
            <div className="w-14 h-14 rounded-2xl bg-navy-900/60 border border-navy-800/80 flex items-center justify-center">
              <ImageIcon className="w-6 h-6 text-muted-foreground/40" />
            </div>
            <p className="text-sm font-semibold text-muted-foreground/60">No results found</p>
            <p className="text-xs text-muted-foreground/40">Try adjusting your search or filters</p>
            <Button variant="outline" size="sm" onClick={clearFilters} className="rounded-xl border-navy-700 text-xs">
              Clear Filters
            </Button>
          </motion.div>
        ) : view === "grid" ? (
          <motion.div
            key="grid"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
          >
            <AnimatePresence>
              {filtered.map((entry, idx) => (
                <motion.div
                  key={entry.id}
                  layout
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: idx * 0.04, duration: 0.3 }}
                >
                  <GridCard entry={entry} onView={handleView} onDelete={() => handleDelete(entry.id)} />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        ) : (
          <motion.div
            key="list"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="space-y-3"
          >
            <AnimatePresence>
              {filtered.map((entry, idx) => (
                <motion.div
                  key={entry.id}
                  layout
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -12 }}
                  transition={{ delay: idx * 0.04, duration: 0.3 }}
                >
                  <ListRow entry={entry} onView={handleView} onDelete={() => handleDelete(entry.id)} />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
export default History
