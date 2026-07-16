import React from "react"
import { useLocation, useNavigate } from "react-router-dom"
import { Bell, Search, Settings, ShieldCheck } from "lucide-react"
import { Button } from "@/components/ui/button"

export const TopNavbar: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()

  // Dynamic route name formatting
  const getPageTitle = (pathname: string) => {
    switch (pathname) {
      case "/":
        return "Dashboard"
      case "/dashboard":
        return "Dashboard"
      case "/upload":
        return "Upload Image"
      case "/prediction-result":
        return "Prediction Result"
      case "/history":
        return "History Log"
      case "/analytics":
        return "Analytics"
      case "/reports":
        return "Reports"
      case "/about":
        return "About Project"
      case "/profile":
        return "Operator Profile"
      default:
        return "DeepSea Console"
    }
  }

  return (
    <header className="flex items-center justify-between h-16 px-6 border-b border-navy-800/80 bg-navy-950/30 backdrop-blur-xl shrink-0 select-none z-10">
      {/* Dynamic Header Name */}
      <div>
        <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-white to-cyan-200 bg-clip-text text-transparent">
          {getPageTitle(location.pathname)}
        </h1>
      </div>

      {/* Action Utilities (Search, Alerts, Profile) */}
      <div className="flex items-center gap-3">
        {/* Quick Search - Collapses on mobile */}
        <div className="relative w-48 sm:w-64 md:block hidden">
          <Search className="absolute left-3 top-2.5 h-4.5 w-4.5 text-muted-foreground/80" />
          <input
            type="text"
            placeholder="Search images..."
            className="w-full h-10 pl-10 pr-4 text-sm bg-navy-900/50 rounded-xl border border-navy-800 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10 outline-none text-foreground placeholder:text-muted-foreground/60 transition-all duration-200"
          />
        </div>

        {/* System Health Badge - Hidden on small mobile */}
        <div className="hidden xs:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-teal-500/5 border border-teal-500/20 text-xs font-semibold text-teal-400">
          <ShieldCheck className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">System Online</span>
        </div>

        {/* Settings Button */}
        <Button
          variant="ghost"
          size="icon"
          className="text-muted-foreground hover:text-cyan-400 hover:bg-navy-800/40 rounded-xl transition-all duration-200 active:scale-95"
          onClick={() => navigate("/profile")}
          title="Operator Settings"
        >
          <Settings className="h-5 w-5" />
        </Button>

        {/* Notifications Alert Bell */}
        <Button
          variant="ghost"
          size="icon"
          className="relative text-muted-foreground hover:text-cyan-400 hover:bg-navy-800/40 rounded-xl transition-all duration-200 active:scale-95"
        >
          <Bell className="h-5 w-5" />
          <span className="absolute top-2 right-2.5 w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="absolute top-2 right-2.5 w-2 h-2 rounded-full bg-cyan-400" />
        </Button>

        {/* Profile Avatar Trigger & Username */}
        <div className="w-[1px] bg-navy-800/80 h-6 mx-1" />
        
        <button
          className="flex items-center gap-2.5 outline-none hover:opacity-90 transition-opacity"
          onClick={() => navigate("/profile")}
        >
          <span className="text-xs font-semibold text-foreground/90 hidden md:block">
            DeepSea Operator
          </span>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-400 to-teal-400 flex items-center justify-center font-bold text-sm text-navy-950 shadow-glow-sm hover:scale-105 active:scale-95 transition-all duration-200">
            DO
          </div>
        </button>
      </div>
    </header>
  )
}
export default TopNavbar
