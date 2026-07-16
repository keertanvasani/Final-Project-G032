import React from "react"
import { Waves, Heart, Shield, Code, Target } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"

export const About: React.FC = () => {
  const techStack = [
    { name: "React 19", role: "UI Component Architecture" },
    { name: "TypeScript", role: "Static Type Security & Schema Safety" },
    { name: "Tailwind CSS", role: "Utility styling & Custom Theme tokenization" },
    { name: "shadcn/ui", role: "Atomic Reusable Elements (Radix primitives)" },
    { name: "Framer Motion", role: "Smooth transitions & collapsible navigations" },
    { name: "Recharts", role: "Responsive image analytical graphics" }
  ]

  return (
    <div className="space-y-6">
      {/* Project Mission Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/25 flex items-center justify-center shadow-glow-sm/5">
              <Waves className="w-5 h-5" />
            </div>
            <div>
              <CardTitle>Project DeepSea AI</CardTitle>
              <CardDescription>Deep sea image classification & marine conservation</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-muted-foreground/80">
          <p>
            DeepSea AI is an advanced computer vision platform designed to connect marine biologists with machine learning classification models. By parsing high-resolution images collected from submersible camera networks, the portal provides immediate identification of marine mammal species, underwater flora, and subsea ecological structures.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 space-y-2 hover:border-cyan-500/20 transition-all duration-300">
              <Heart className="w-5 h-5 text-rose-400" />
              <h4 className="font-bold text-foreground text-xs uppercase tracking-wider">Cetacean Conservation</h4>
              <p className="text-[11px] text-muted-foreground">Identifying marine mammal positions in camera feeds to prevent vessel collisions and evaluate pod migrations.</p>
            </div>
            <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 space-y-2 hover:border-cyan-500/20 transition-all duration-300">
              <Shield className="w-5 h-5 text-cyan-400" />
              <h4 className="font-bold text-foreground text-xs uppercase tracking-wider">Habitat Mapping</h4>
              <p className="text-[11px] text-muted-foreground">Logging coral reef distributions, benthic organisms, and seabed geological layouts over time.</p>
            </div>
            <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 space-y-2 hover:border-cyan-500/20 transition-all duration-300">
              <Target className="w-5 h-5 text-teal-400" />
              <h4 className="font-bold text-foreground text-xs uppercase tracking-wider">AI Image Classification</h4>
              <p className="text-[11px] text-muted-foreground">Leveraging convolutional neural networks to catalog marine species across subsea camera networks.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tech Stack Details */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="w-5 h-5 text-cyan-400" />
            System Technology Stack
          </CardTitle>
          <CardDescription>Built with modern libraries for performance and rich aesthetics.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {techStack.map((tech, idx) => (
              <div key={idx} className="p-3.5 rounded-xl bg-navy-950/40 border border-navy-800/80 flex flex-col justify-center hover:border-cyan-500/25 hover:shadow-glow-sm/5 transition-all duration-250">
                <span className="text-sm font-bold text-foreground">{tech.name}</span>
                <span className="text-[10px] text-muted-foreground mt-0.5">{tech.role}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
export default About
