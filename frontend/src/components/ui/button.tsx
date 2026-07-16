import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98] [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-gradient-to-r from-cyan-400 to-teal-400 text-navy-950 shadow-glow-sm hover:shadow-glow-md hover:brightness-110 active:brightness-95 border border-cyan-300/20",
        destructive:
          "bg-rose-500/10 text-rose-400 border border-rose-500/20 hover:bg-rose-500/20 hover:text-rose-300 shadow-sm",
        outline:
          "border border-navy-700 bg-navy-950/40 backdrop-blur-md text-foreground hover:bg-navy-800 hover:border-cyan-500/30 hover:text-cyan-400 shadow-sm",
        secondary:
          "bg-navy-800/80 text-cyan-400 border border-navy-700 hover:bg-navy-700 hover:border-cyan-500/30 hover:text-cyan-300 hover:shadow-glow-sm/10 shadow-sm",
        ghost: "text-muted-foreground hover:text-cyan-400 hover:bg-navy-800/50",
        link: "text-cyan-400 underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-5 rounded-[14px]",
        sm: "h-8 px-3 rounded-xl text-xs",
        lg: "h-12 px-8 rounded-2xl text-base",
        icon: "h-10 w-10 rounded-[14px]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
