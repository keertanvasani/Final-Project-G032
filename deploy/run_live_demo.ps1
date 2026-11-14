<#
  Start the live backend and expose it through a temporary Cloudflare tunnel.

  Use this only for the review itself. It makes this laptop reachable from the
  internet through a random URL for as long as the script runs; Ctrl+C closes the
  tunnel and stops the server, and the URL stops working immediately.

  The static Space is always up and shows recorded results. This script adds the
  live half: it prints a link that opens the same page wired to this laptop.

  Usage:
    .\deploy\run_live_demo.ps1 -SpaceUrl https://YOUR-USER-YOUR-SPACE.static.hf.space
#>
param(
  [string]$SpaceUrl = "",
  [int]$Port = 8000,
  [int]$TimeoutSec = 240
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root "venv\Scripts\python.exe"
$cf   = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
if (-not (Test-Path $cf)) { $cf = (Get-Command cloudflared -ErrorAction Stop).Source }

$tmp = Join-Path $env:TEMP "deepsea_live"
New-Item -ItemType Directory -Force $tmp | Out-Null
$srvLog = Join-Path $tmp "server.log"; $tunLog = Join-Path $tmp "tunnel.log"
Remove-Item $srvLog, $tunLog -ErrorAction SilentlyContinue

# Keep the machine awake while this runs: a sleeping laptop drops the tunnel.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
# The flags are [uint32] casts of decimal values on purpose: PowerShell reads the hex
# literal 0x80000001 as a negative Int32, which the API rejects, and with
# $ErrorActionPreference = "Stop" that would abort the whole script.
[uint32]$ES_CONTINUOUS = 2147483648                                    # 0x80000000
[uint32]$ES_HOLD       = $ES_CONTINUOUS -bor 1                         # | ES_SYSTEM_REQUIRED
[void][Win32.Power]::SetThreadExecutionState($ES_HOLD)

$server = $null; $tunnel = $null
try {
  # Reuse a server that is already healthy on this port rather than fighting it.
  $up = $false
  try { $up = (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).models_loaded } catch {}
  if (-not $up) {
    Write-Host "Starting backend on port $Port ..."
    $server = Start-Process -FilePath $py -WorkingDirectory $root -PassThru -WindowStyle Hidden `
      -ArgumentList "-m","uvicorn","webapp:app","--host","127.0.0.1","--port","$Port" `
      -RedirectStandardOutput $srvLog -RedirectStandardError "$srvLog.err"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline -and -not $up) {
      Start-Sleep -Seconds 3
      try { $up = (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).models_loaded } catch {}
    }
    if (-not $up) { throw "Backend did not become healthy within $TimeoutSec s. See $srvLog.err" }
  }
  $h = Invoke-RestMethod "http://127.0.0.1:$Port/health"
  Write-Host "Backend healthy (device: $($h.device))."

  Write-Host "Opening tunnel ..."
  $tunnel = Start-Process -FilePath $cf -PassThru -WindowStyle Hidden `
    -ArgumentList "tunnel","--url","http://127.0.0.1:$Port","--no-autoupdate" `
    -RedirectStandardError $tunLog -RedirectStandardOutput "$tunLog.out"
  $url = $null
  $deadline = (Get-Date).AddSeconds(60)
  while ((Get-Date) -lt $deadline -and -not $url) {
    Start-Sleep -Seconds 2
    if (Test-Path $tunLog) {
      $m = Select-String -Path $tunLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
      if ($m) { $url = $m.Matches[0].Value }
    }
  }
  if (-not $url) { throw "No tunnel URL appeared. See $tunLog" }

  # The tunnel needs a moment before it routes; confirm it end to end.
  $ok = $false
  for ($i = 0; $i -lt 15 -and -not $ok; $i++) {
    try { $ok = (Invoke-RestMethod "$url/health" -TimeoutSec 8).models_loaded } catch { Start-Sleep -Seconds 2 }
  }
  if (-not $ok) { throw "Tunnel opened but $url/health did not answer." }

  Write-Host ""
  Write-Host "LIVE BACKEND : $url" -ForegroundColor Green
  Write-Host "Full app     : $url/   (this laptop serves the whole site)"
  if ($SpaceUrl) {
    Write-Host "Static + live: $($SpaceUrl.TrimEnd('/'))/?api=$url" -ForegroundColor Green
  } else {
    Write-Host "Static + live: <your-space-url>/?api=$url"
  }
  Write-Host ""
  Write-Host "Anyone with these links can use your GPU. Press Ctrl+C to close them."
  Wait-Process -Id $tunnel.Id
}
finally {
  [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)   # release the awake request
  foreach ($proc in @($tunnel, $server)) {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
  }
  Write-Host "Tunnel and server stopped."
}
