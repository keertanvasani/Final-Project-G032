import React from "react"
import { useNavigate } from "react-router-dom"
import { Waves, Layers, Target, Award, ShieldAlert } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card"
import { AnimatedBackground } from "@/components/layout/AnimatedBackground"
import underwaterImg from "@/assets/underwater.png"

export const Login: React.FC = () => {
  const navigate = useNavigate()

  const handleGoogleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    navigate("/")
  }

  const features = [
    {
      title: "Ensemble Detection",
      description: "Combines multiple deep convolutional architectures to map marine biodiversity.",
      icon: Layers,
      color: "text-cyan-400"
    },
    {
      title: "High Accuracy",
      description: "Reaches 98.4% detection marks on complex deep-sea imagery classifications.",
      icon: Target,
      color: "text-teal-400"
    },
    {
      title: "Research Grade Results",
      description: "Meets international bioecological dataset classification compliance.",
      icon: Award,
      color: "text-cyan-300"
    }
  ]

  return (
    <div className="flex min-h-screen w-screen bg-[#071A2E] text-foreground font-sans overflow-hidden select-none">
      
      {/* ── LEFT SIDE: Cover Image & Feature Cards (Desktop Only) ── */}
      <div className="hidden lg:flex lg:w-1/2 relative flex-col justify-between p-12 overflow-hidden border-r border-navy-800/50">
        {/* Undersea background turtle photo */}
        <img
          src={underwaterImg}
          alt="Deep sea turtle and coral reef"
          className="absolute inset-0 w-full h-full object-cover object-center pointer-events-none scale-105"
        />
        
        {/* Ocean gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-[#071A2E] via-[#071A2E]/70 to-[#071A2E]/30" />
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-[#071A2E]/90" />
        
        {/* Ambient bioluminescent glows */}
        <div className="absolute top-1/4 left-1/4 w-80 h-80 rounded-full bg-cyan-500/10 blur-[120px] pointer-events-none" />

        {/* Top Header branding */}
        <div className="relative z-10 space-y-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-2xl bg-cyan-500/15 text-cyan-450 text-cyan-400 border border-cyan-500/25 shadow-glow-sm/5">
              <Waves className="w-6 h-6 animate-pulse" />
            </div>
            <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-white to-cyan-300 bg-clip-text text-transparent">
              DeepSea AI
            </span>
          </div>
          <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white leading-tight">
            AI-powered Deep Sea Image <br />
            Classification & Detection
          </h2>
        </div>

        {/* Middle Feature badges */}
        <div className="relative z-10 space-y-4 my-8 max-w-md">
          {features.map((f, idx) => (
            <div
              key={idx}
              className="p-4 rounded-2xl bg-navy-950/45 backdrop-blur-md border border-navy-800/40 hover:border-cyan-500/20 hover:shadow-glow-sm/5 transition-all duration-300 flex items-start gap-4"
            >
              <div className="mt-0.5">
                <f.icon className={`w-5 h-5 ${f.color}`} />
              </div>
              <div className="space-y-0.5">
                <h4 className="text-xs font-bold uppercase tracking-wider text-foreground">{f.title}</h4>
                <p className="text-xs text-muted-foreground/80 leading-relaxed">{f.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom Academic info */}
        <div className="relative z-10 border-t border-navy-800/40 pt-6">
          <div className="text-xs font-bold text-foreground/80 tracking-wide">
            Amrita Vishwa Vidyapeetham
          </div>
          <div className="text-[10px] text-muted-foreground/70 uppercase tracking-widest mt-1">
            Final Year Project • 2026
          </div>
        </div>
      </div>

      {/* ── RIGHT SIDE: minimal elegant login card ── */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 relative">
        {/* Animated fluid backdrops */}
        <AnimatedBackground />

        <Card className="w-full max-w-md border-navy-800 bg-navy-900/60 backdrop-blur-2xl shadow-card relative z-10 rounded-2xl hover:border-cyan-500/20 hover:shadow-card-hover transition-all duration-300 p-2">
          <CardHeader className="space-y-2 text-center pb-8 pt-6">
            <div className="flex items-center justify-center mx-auto w-11 h-11 rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-2 sm:hidden shadow-glow-sm/5">
              <Waves className="w-5.5 h-5.5" />
            </div>
            <CardTitle className="text-2xl font-bold tracking-tight text-white">
              Welcome Back
            </CardTitle>
            <CardDescription className="text-muted-foreground/80 text-sm">
              Continue using your Google Account
            </CardDescription>
          </CardHeader>
          <CardContent className="pb-6">
            {/* Google Authentication Button */}
            <button
              onClick={handleGoogleLogin}
              className="w-full h-12 flex items-center justify-center gap-3 bg-navy-950 border border-navy-800 hover:border-cyan-500/50 hover:bg-navy-900 rounded-xl text-sm font-semibold text-foreground transition-all duration-200 shadow-sm active:scale-[0.98] outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background group"
            >
              {/* Google logo SVG */}
              <svg className="w-5 h-5 shrink-0 transition-transform group-hover:scale-105" viewBox="0 0 24 24">
                <path
                  fill="#EA4335"
                  d="M12.24 10.285V14.4h6.887c-.648 2.41-2.519 4.114-5.136 4.114-3.555 0-6.435-2.88-6.435-6.435s2.88-6.435 6.435-6.435c1.637 0 3.136.612 4.3 1.62l3.053-3.053C18.847 2.052 15.688 1 12.24 1 6.043 1 1 6.043 1 12.24s5.043 11.24 11.24 11.24c5.898 0 10.742-4.257 10.742-10.742 0-.693-.06-1.373-.173-2.033H12.24z"
                />
              </svg>
              Sign in with Google
            </button>
          </CardContent>
          <CardFooter className="flex flex-col space-y-2 text-center text-[10px] uppercase tracking-wider text-muted-foreground/60 border-t border-navy-800/30 pt-6 pb-4">
            <div className="flex items-center gap-1.5 justify-center">
              <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
              <span>Secure Encrypted Portal</span>
            </div>
            {/* Mobile context */}
            <span className="lg:hidden block lowercase tracking-normal text-muted-foreground/50">
              Amrita Vishwa Vidyapeetham • FYP 2026
            </span>
          </CardFooter>
        </Card>
      </div>

    </div>
  )
}
export default Login
