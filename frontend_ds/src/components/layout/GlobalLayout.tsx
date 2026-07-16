import React from "react"
import { Outlet, useLocation } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { Sidebar } from "./Sidebar"
import { TopNavbar } from "./TopNavbar"
import { AnimatedBackground } from "./AnimatedBackground"

export const GlobalLayout: React.FC = () => {
  const location = useLocation()

  return (
    <div className="flex h-screen w-screen overflow-hidden text-foreground" style={{ background: "#071A2E" }}>
      {/* ── Full-screen animated ocean background (z-0) ── */}
      <AnimatedBackground />

      {/* ── Sidebar (z-20) ── */}
      <Sidebar />

      {/* ── Main content column ── */}
      <div className="flex flex-col flex-1 h-full overflow-hidden relative" style={{ zIndex: 10 }}>
        {/* Top navigation */}
        <TopNavbar />

        {/* Scrollable page viewport */}
        <main className="flex-1 overflow-y-auto relative">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.2, ease: "easeInOut" }}
                className="space-y-6"
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  )
}

export default GlobalLayout
