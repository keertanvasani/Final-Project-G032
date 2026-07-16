import React from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { ThemeProvider } from "@/context/ThemeContext"
import { GlobalLayout } from "@/components/layout/GlobalLayout"

// Import placeholder pages
import { Login } from "@/pages/Login"
import { Dashboard } from "@/pages/Dashboard"
import { Upload } from "@/pages/Upload"
import { PredictionResult } from "@/pages/PredictionResult"
import { History } from "@/pages/History"
import { Analytics } from "@/pages/Analytics"
import { Reports } from "@/pages/Reports"
import { About } from "@/pages/About"
import { Profile } from "@/pages/Profile"

export const App: React.FC = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          {/* Public authentication route */}
          <Route path="/login" element={<Login />} />

          {/* Core application routes encapsulated inside the GlobalLayout */}
          <Route element={<GlobalLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Navigate to="/" replace />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/prediction-result" element={<PredictionResult />} />
            <Route path="/history" element={<History />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/about" element={<About />} />
            <Route path="/profile" element={<Profile />} />
          </Route>

          {/* Catch-all fallback redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}
export default App
