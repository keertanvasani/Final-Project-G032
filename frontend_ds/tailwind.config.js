/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: { "2xl": "1400px" },
    },
    extend: {
      colors: {
        // Core semantic tokens
        border:      "hsl(var(--border))",
        input:       "hsl(var(--input))",
        ring:        "hsl(var(--ring))",
        background:  "hsl(var(--background))",
        foreground:  "hsl(var(--foreground))",

        // Ocean palette
        navy: {
          950: "#030d1a",
          900: "#071A2E",
          800: "#0d2440",
          700: "#103356",
          600: "#154a7a",
        },
        cyan: {
          glow:   "#67E8F9",
          400:    "#22D3EE",
          500:    "#06B6D4",
          600:    "#0891B2",
        },
        teal: {
          400:    "#2DD4BF",
          500:    "#14B8A6",
          600:    "#0D9488",
        },
        bio: {
          cyan:   "rgba(34, 211, 238, 0.15)",
          teal:   "rgba(20, 184, 166, 0.15)",
          glow:   "rgba(103, 232, 249, 0.08)",
        },

        // shadcn tokens
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
      },

      borderRadius: {
        "4xl": "2rem",
        "3xl": "1.5rem",
        "2xl": "1.25rem",
        xl:    "1rem",
        lg:    "var(--radius)",
        md:    "calc(var(--radius) - 2px)",
        sm:    "calc(var(--radius) - 4px)",
      },

      boxShadow: {
        "glow-sm":  "0 0 12px rgba(34, 211, 238, 0.20)",
        "glow-md":  "0 0 28px rgba(34, 211, 238, 0.25)",
        "glow-lg":  "0 0 60px rgba(34, 211, 238, 0.18)",
        "glow-teal":"0 0 28px rgba(20, 184, 166, 0.22)",
        "deep":     "0 8px 32px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.3)",
        "card":     "0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)",
        "card-hover": "0 8px 40px rgba(0,0,0,0.5), 0 0 20px rgba(34,211,238,0.12), inset 0 1px 0 rgba(255,255,255,0.07)",
      },

      backgroundImage: {
        "ocean-gradient":  "linear-gradient(135deg, #071A2E 0%, #0d2440 50%, #071A2E 100%)",
        "card-gradient":   "linear-gradient(135deg, rgba(13,36,64,0.8) 0%, rgba(7,26,46,0.6) 100%)",
        "primary-gradient":"linear-gradient(135deg, #0EA5E9 0%, #06B6D4 50%, #14B8A6 100%)",
        "bio-gradient":    "linear-gradient(135deg, rgba(34,211,238,0.15) 0%, rgba(20,184,166,0.08) 100%)",
      },

      keyframes: {
        "float-slow": {
          "0%, 100%": { transform: "translateY(0px) translateX(0px) scale(1)" },
          "33%":       { transform: "translateY(-30px) translateX(15px) scale(1.05)" },
          "66%":       { transform: "translateY(20px) translateX(-10px) scale(0.97)" },
        },
        "float-medium": {
          "0%, 100%": { transform: "translateY(0px) translateX(0px) scale(1)" },
          "40%":       { transform: "translateY(25px) translateX(-20px) scale(1.04)" },
          "70%":       { transform: "translateY(-15px) translateX(25px) scale(0.96)" },
        },
        "float-fast": {
          "0%, 100%": { transform: "translateY(0px) translateX(0px)" },
          "50%":       { transform: "translateY(-20px) translateX(10px)" },
        },
        "particle-drift": {
          "0%":   { transform: "translateY(100vh) translateX(0px)", opacity: "0" },
          "10%":  { opacity: "1" },
          "90%":  { opacity: "0.6" },
          "100%": { transform: "translateY(-10vh) translateX(40px)", opacity: "0" },
        },
        "pulse-glow": {
          "0%, 100%": { opacity: "0.4", transform: "scale(1)" },
          "50%":       { opacity: "0.8", transform: "scale(1.08)" },
        },
        "shimmer": {
          "0%":   { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" },
        },
        "accordion-down": {
          from: { height: "0" },
          to:   { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to:   { height: "0" },
        },
      },
      animation: {
        "float-slow":    "float-slow 18s ease-in-out infinite",
        "float-medium":  "float-medium 13s ease-in-out infinite",
        "float-fast":    "float-fast 9s ease-in-out infinite",
        "pulse-glow":    "pulse-glow 4s ease-in-out infinite",
        "particle-drift":"particle-drift 12s linear infinite",
        "accordion-down":"accordion-down 0.2s ease-out",
        "accordion-up":  "accordion-up 0.2s ease-out",
      },

      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
