import React from "react"
import { Shield, Key, Mail, Calendar, Settings } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export const Profile: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Profile Overview Card */}
        <Card className="lg:col-span-1">
          <CardHeader className="text-center pb-6">
            <div className="relative mx-auto w-24 h-24 rounded-2xl bg-gradient-to-tr from-cyan-400 to-teal-400 p-[2px] shadow-glow-md">
              <div className="w-full h-full rounded-2xl bg-navy-950 flex items-center justify-center font-bold text-2xl text-cyan-400">
                DS
              </div>
            </div>
            <h2 className="text-lg font-bold text-foreground mt-4">DeepSea Operator</h2>
            <p className="text-xs text-cyan-400/90 font-mono mt-0.5">operator@deepsea.ai</p>
            <span className="inline-flex items-center gap-1 mt-3 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Shield className="w-3 h-3" />
              Level-3 Operator
            </span>
          </CardHeader>
          <CardContent className="space-y-4 pt-4 border-t border-navy-800/40 text-sm">
            <div className="flex justify-between items-center text-xs">
              <span className="text-muted-foreground">Duty Station</span>
              <span className="font-semibold text-foreground">Pacific Northwest Basin</span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-muted-foreground">Session Uptime</span>
              <span className="font-semibold text-foreground">04h 32m</span>
            </div>
          </CardContent>
        </Card>

        {/* Right Credentials Management Card */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Operator Security Details</CardTitle>
            <CardDescription>Manage security keys and image classification access layers.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-xl bg-navy-950/40 border border-navy-800/80 space-y-4">
              <div className="flex items-center gap-3">
                <Mail className="w-5 h-5 text-cyan-400" />
                <div className="flex-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Access Channel</div>
                  <div className="text-sm font-semibold text-foreground">operator@deepsea.ai</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <Key className="w-5 h-5 text-teal-400" />
                <div className="flex-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Security Token</div>
                  <div className="text-sm font-mono text-foreground">ds_operator_sha256_e8d9...</div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Calendar className="w-5 h-5 text-cyan-400" />
                <div className="flex-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Last Re-auth</div>
                  <div className="text-sm font-semibold text-foreground">2026-07-13 18:12 UT</div>
                </div>
              </div>
            </div>
            
            <div className="flex gap-3 justify-end pt-2">
              <Button variant="outline" size="sm" className="rounded-xl border-navy-800 text-xs">
                Rotate Security Token
              </Button>
              <Button size="sm" className="rounded-xl text-xs font-bold">
                <Settings className="w-3.5 h-3.5 mr-1.5" />
                Profile Settings
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
export default Profile
