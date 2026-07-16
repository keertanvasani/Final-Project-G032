import React, { useState } from "react"
import { NavLink, useLocation } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import {
  LayoutDashboard,
  Image,
  History,
  BarChart3,
  FileText,
  Info,
  User,
  LogOut,
  Waves,
  ChevronLeft,
  ChevronRight
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface SidebarProps {
  className?: string
}

export const Sidebar: React.FC<SidebarProps> = ({ className }) => {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const location = useLocation()

  const navItems = [
    { name: "Dashboard", path: "/", icon: LayoutDashboard },
    { name: "Upload Image", path: "/upload", icon: Image },
    { name: "History", path: "/history", icon: History },
    { name: "Analytics", path: "/analytics", icon: BarChart3 },
    { name: "Reports", path: "/reports", icon: FileText },
    { name: "About Project", path: "/about", icon: Info },
    { name: "Profile", path: "/profile", icon: User },
  ]

  return (
    <motion.aside
      initial={{ width: 260 }}
      animate={{ width: isCollapsed ? 76 : 260 }}
      transition={{ duration: 0.3, ease: [0.25, 0.8, 0.25, 1] }}
      className={cn(
        "relative flex flex-col h-screen border-r border-navy-800/80 bg-navy-950/45 backdrop-blur-2xl text-foreground select-none shrink-0 z-20",
        className
      )}
    >
      {/* Header / Logo */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-navy-850/60">
        <NavLink to="/" className="flex items-center gap-3 overflow-hidden">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/25 shrink-0 shadow-glow-sm/5">
            <Waves className="w-5 h-5 animate-pulse" />
          </div>
          <AnimatePresence>
            {!isCollapsed && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="font-bold text-base tracking-tight bg-gradient-to-r from-cyan-400 via-cyan-300 to-teal-400 bg-clip-text text-transparent whitespace-nowrap"
              >
                DeepSea AI
              </motion.span>
            )}
          </AnimatePresence>
        </NavLink>
        
        {/* Collapse Button */}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="absolute -right-3 top-5 w-6 h-6 rounded-full border border-navy-700 bg-navy-900 shadow-deep hover:bg-navy-800 hover:text-cyan-400 z-30 transition-transform duration-200 active:scale-90"
        >
          {isCollapsed ? (
            <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
          ) : (
            <ChevronLeft className="w-3.5 h-3.5 text-muted-foreground" />
          )}
        </Button>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto overflow-x-hidden">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 group relative border border-transparent",
                  isActive
                    ? "bg-gradient-to-r from-cyan-500/12 to-teal-500/8 text-cyan-400 border-cyan-500/20 shadow-glow-sm"
                    : "text-muted-foreground hover:bg-navy-800/40 hover:text-foreground hover:translate-x-0.5"
                )
              }
            >
              <item.icon
                className={cn(
                  "w-5 h-5 shrink-0 transition-transform duration-300 group-hover:scale-105",
                  isActive ? "text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]" : "text-muted-foreground group-hover:text-foreground"
                )}
              />
              {!isCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="whitespace-nowrap transition-colors"
                >
                  {item.name}
                </motion.span>
              )}
              {/* Highlight active glowing strip */}
              {isActive && (
                <motion.div
                  layoutId="active-strip"
                  className="absolute left-0 top-2.5 bottom-2.5 w-1 rounded-r-full bg-cyan-400 shadow-[0_0_10px_#22d3ee]"
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer / Logout */}
      <div className="p-3 border-t border-navy-850/60 bg-navy-950/20">
        <NavLink
          to="/login"
          className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold text-rose-450 hover:bg-rose-500/10 hover:text-rose-450 transition-all duration-200 group"
        >
          <LogOut className="w-5 h-5 shrink-0 transition-transform group-hover:translate-x-[-2px]" />
          {!isCollapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="whitespace-nowrap"
            >
              Logout
            </motion.span>
          )}
        </NavLink>
      </div>
    </motion.aside>
  )
}
