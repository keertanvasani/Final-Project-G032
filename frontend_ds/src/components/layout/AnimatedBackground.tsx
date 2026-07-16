import React, { useEffect, useRef } from "react"

/* ─── Particle Canvas ──────────────────────────────────────────────────── */
const ParticleCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animId: number
    const particles: {
      x: number; y: number; r: number
      vx: number; vy: number; opacity: number
      color: string
    }[] = []

    const COLORS = [
      "rgba(34, 211, 238, VAL)",
      "rgba(20, 184, 166, VAL)",
      "rgba(103, 232, 249, VAL)",
      "rgba(14, 165, 233, VAL)",
    ]

    const resize = () => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }

    const spawn = () => {
      const color = COLORS[Math.floor(Math.random() * COLORS.length)]
      particles.push({
        x:       Math.random() * canvas.width,
        y:       canvas.height + 10,
        r:       Math.random() * 1.8 + 0.4,
        vx:      (Math.random() - 0.5) * 0.35,
        vy:      -(Math.random() * 0.5 + 0.2),
        opacity: Math.random() * 0.6 + 0.1,
        color,
      })
    }

    let spawnTimer = 0
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      spawnTimer++
      if (spawnTimer % 4 === 0 && particles.length < 120) spawn()

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.x += p.vx
        p.y += p.vy
        p.opacity -= 0.0008

        if (p.y < -10 || p.opacity <= 0) {
          particles.splice(i, 1)
          continue
        }

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = p.color.replace("VAL", p.opacity.toString())
        ctx.fill()
      }
      animId = requestAnimationFrame(draw)
    }

    resize()
    draw()
    window.addEventListener("resize", resize)
    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener("resize", resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 1 }}
    />
  )
}

/* ─── Floating Orb ──────────────────────────────────────────────────────── */
interface OrbProps {
  className?: string
  color: string
  size: number
  animation: string
  delay?: number
  opacity?: number
  blur?: number
}

const FloatingOrb: React.FC<OrbProps> = ({
  className = "",
  color,
  size,
  animation,
  delay = 0,
  opacity = 0.18,
  blur = 80,
}) => (
  <div
    className={`absolute rounded-full pointer-events-none ${animation} ${className}`}
    style={{
      width:     size,
      height:    size,
      background: color,
      opacity,
      filter:    `blur(${blur}px)`,
      animationDelay: `${delay}s`,
      willChange: "transform",
    }}
  />
)

/* ─── Main AnimatedBackground ───────────────────────────────────────────── */
export const AnimatedBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
      {/* ── Base ocean gradient fill ── */}
      <div
        className="absolute inset-0"
        style={{
          background: `
            radial-gradient(ellipse 80% 60% at 50% -10%, rgba(7,200,216,0.12) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(20,184,166,0.10) 0%, transparent 60%),
            radial-gradient(ellipse 50% 50% at 0% 80%,   rgba(14,165,233,0.08) 0%, transparent 60%),
            linear-gradient(180deg, #030d1a 0%, #071A2E 35%, #0a1f36 70%, #071A2E 100%)
          `,
        }}
      />

      {/* ── Floating blurred orbs ── */}
      {/* Top-left large cyan orb */}
      <FloatingOrb
        className="top-[-15%] left-[-10%]"
        color="radial-gradient(circle, #22D3EE, #0EA5E9)"
        size={600}
        animation="animate-float-slow"
        delay={0}
        opacity={0.15}
        blur={100}
      />
      {/* Bottom-right teal orb */}
      <FloatingOrb
        className="bottom-[-20%] right-[-8%]"
        color="radial-gradient(circle, #14B8A6, #0D9488)"
        size={550}
        animation="animate-float-medium"
        delay={4}
        opacity={0.13}
        blur={110}
      />
      {/* Centre accent orb */}
      <FloatingOrb
        className="top-[30%] left-[40%]"
        color="radial-gradient(circle, #06B6D4, #0EA5E9)"
        size={400}
        animation="animate-float-fast"
        delay={7}
        opacity={0.07}
        blur={90}
      />
      {/* Top-right secondary orb */}
      <FloatingOrb
        className="top-[5%] right-[10%]"
        color="radial-gradient(circle, #67E8F9, #22D3EE)"
        size={320}
        animation="animate-float-medium"
        delay={2}
        opacity={0.10}
        blur={85}
      />
      {/* Bottom-left accent */}
      <FloatingOrb
        className="bottom-[5%] left-[15%]"
        color="radial-gradient(circle, #2DD4BF, #14B8A6)"
        size={260}
        animation="animate-float-slow"
        delay={9}
        opacity={0.09}
        blur={75}
      />

      {/* ── Horizontal glow band (horizon effect) ── */}
      <div
        className="absolute left-0 right-0"
        style={{
          top:    "62%",
          height: "1px",
          background: "linear-gradient(90deg, transparent 0%, rgba(34,211,238,0.18) 30%, rgba(20,184,166,0.22) 50%, rgba(34,211,238,0.18) 70%, transparent 100%)",
          filter: "blur(1px)",
          boxShadow: "0 0 40px 6px rgba(34,211,238,0.08)",
        }}
      />

      {/* ── Subtle grid overlay ── */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(rgba(34,211,238,0.022) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34,211,238,0.022) 1px, transparent 1px)
          `,
          backgroundSize: "48px 48px",
        }}
      />

      {/* ── Rising particles canvas ── */}
      <ParticleCanvas />

      {/* ── Vignette edge darkening ── */}
      <div
        className="absolute inset-0"
        style={{
          background: `
            radial-gradient(ellipse 100% 100% at 50% 50%,
              transparent 40%,
              rgba(3,13,26,0.45) 100%
            )
          `,
        }}
      />
    </div>
  )
}
