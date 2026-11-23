#Requires -Version 5.1
<#
.SYNOPSIS
  One-command launcher for InkEngine (headless background server) + OpenCode CLI.
  Usage:
    inkcode [opencode args...]                 Launch OpenCode CLI with Inkling-2-Mini (auto-starts background server)
    inkcode --ctx 131072 [opencode args...]    Set context length (65536 | 131072 | 262144 | 524288)
    inkcode --effort max [opencode args...]    Set thinking effort (none | minimal | low | medium | high | xhigh | max)
    inkcode start [--ctx N] [--effort L]       Start background server only (no TUI)
    inkcode status                             Show live VRAM / RAM / SSD telemetry
    inkcode stop                               Shut down background server & reclaim all VRAM/RAM
    inkcode restart [--ctx N] [--effort L]     Restart background server with new settings
#>

$EngineDir  = $PSScriptRoot
$EngineExe  = Join-Path $EngineDir "build\inkengine.exe"
$LogDir     = Join-Path $EngineDir "logs"
$OutLog     = Join-Path $LogDir "server.log"
$ErrLog     = Join-Path $LogDir "server.err.log"
$Port       = 8787
$BaseUrl    = "http://127.0.0.1:$Port"
$ModelSpec  = "inkengine/Inkling-2-Mini"

$Action      = "launch"
$Ctx         = 131072
$CtxExplicit = $false
$Effort      = "max"
$EffortExplicit = $false
$RamKvMb     = 1024
$GgufPath    = $env:INKENGINE_MODEL
$MmprojPath  = $env:INKENGINE_MMPROJ
$NoVision    = $false
$PassArgs    = New-Object System.Collections.Generic.List[string]

# Parse inkcode options while preserving all OpenCode CLI arguments intact
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    if ($i -eq 0 -and $a -in @("start", "stop", "status", "restart", "help", "--ink-help")) {
        $Action = $a
        continue
    }
    switch ($a) {
        "--ctx" {
            if ($i + 1 -lt $args.Count) { $Ctx = [int]$args[++$i]; $CtxExplicit = $true }
        }
        "--effort" {
            if ($i + 1 -lt $args.Count) { $Effort = [string]$args[++$i]; $EffortExplicit = $true }
        }
        "--ram-kv-mb" {
            if ($i + 1 -lt $args.Count) { $RamKvMb = [int]$args[++$i] }
        }
        "--gguf" {
            if ($i + 1 -lt $args.Count) { $GgufPath = [string]$args[++$i] }
        }
        "--mmproj" {
            if ($i + 1 -lt $args.Count) { $MmprojPath = [string]$args[++$i] }
        }
        "--no-vision" {
            $NoVision = $true
        }
        default {
            $PassArgs.Add($a)
        }
    }
}

function Get-InkHealth {
    try {
        return Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 2 -ErrorAction Stop
    } catch {
        return $null
    }
}

function Stop-InkServer {
    $h = Get-InkHealth
    if ($h) {
        try {
            Invoke-RestMethod -Uri "$BaseUrl/shutdown" -Method Post -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null
            Start-Sleep -Milliseconds 600
        } catch {}
    }
    $procs = Get-Process -Name "inkengine" -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 400
    }
}

function Start-InkServerBackground {
    param([int]$TargetCtx, [string]$TargetEffort, [int]$TargetRamKv)

    if (-not (Test-Path $EngineExe)) {
        Write-Host "[InkCode] ERROR: Missing $EngineExe. Run build.bat first." -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

    $argStr = "--serve --host 127.0.0.1 --port $Port --ctx $TargetCtx --ram-kv-mb $TargetRamKv --effort $TargetEffort"
    if ($GgufPath) {
        $argStr = "--model `"$GgufPath`" " + $argStr
    }
    if ($MmprojPath) {
        $argStr = "--mmproj `"$MmprojPath`" " + $argStr
    }
    if ($NoVision) {
        $argStr = "--no-vision " + $argStr
    }
    Write-Host "[InkEngine] Loading Inkling-2-Mini in background (ctx=$TargetCtx, effort=$TargetEffort)..." -ForegroundColor Cyan

    $proc = Start-Process -FilePath $EngineExe `
        -ArgumentList $argStr `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    while ($sw.Elapsed.TotalSeconds -lt 120) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) {
            Write-Host "`n[InkEngine] Server exited prematurely (code $($proc.ExitCode)). Check $OutLog / $ErrLog" -ForegroundColor Red
            if (Test-Path $ErrLog) { Get-Content $ErrLog -Tail 15 }
            exit 1
        }
        $h = Get-InkHealth
        if ($h -and $h.status -eq "ok") {
            $ready = $true
            break
        }
        Write-Host -NoNewline ("`r[InkEngine] Warming VRAM/RAM expert cache... {0:N0}s " -f $sw.Elapsed.TotalSeconds)
    }
    if (-not $ready) {
        Write-Host "`n[InkEngine] Timed out waiting for background server." -ForegroundColor Red
        exit 1
    }
    Write-Host ("`r[InkEngine] Ready in {0:N1}s (PID=$($proc.Id), ctx=$TargetCtx, effort=$TargetEffort)          " -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

switch ($Action) {
    "help" {
        Write-Host "InkCode - Lightweight Background InkEngine + OpenCode CLI Launcher" -ForegroundColor Cyan
        Write-Host "  inkcode [project|args...]           Auto-start InkEngine in background & launch OpenCode CLI"
        Write-Host "  inkcode --ctx 65536|131072|262144   Launch with specific context length (default: 131072)"
        Write-Host "  inkcode --effort none|low|med|high|max  Set reasoning effort (default: max)"
        Write-Host "  inkcode --gguf <path>               Use a custom GGUF path (or set INKENGINE_MODEL env var)"
        Write-Host "  inkcode start                       Preload InkEngine in background only"
        Write-Host "  inkcode status                      Show live VRAM / RAM / SSD telemetry"
        Write-Host "  inkcode restart                     Restart background server"
        Write-Host "  inkcode stop                        Stop background server & free all VRAM/RAM"
        exit 0
    }
    "--ink-help" {
        & $PSCommandPath "help"
        exit 0
    }
    "stop" {
        $procs = Get-Process -Name "inkengine" -ErrorAction SilentlyContinue
        if (-not $procs -and -not (Get-InkHealth)) {
            Write-Host "[InkEngine] Not running." -ForegroundColor Yellow
            exit 0
        }
        Stop-InkServer
        Write-Host "[InkEngine] Background server stopped. All VRAM and RAM freed." -ForegroundColor Green
        exit 0
    }
    "status" {
        $h = Get-InkHealth
        if (-not $h) {
            Write-Host "[InkEngine] Status: STOPPED (run 'inkcode' or 'inkcode start' to load)" -ForegroundColor Yellow
            exit 0
        }
        $st = Invoke-RestMethod -Uri "$BaseUrl/stats" -TimeoutSec 3
        $proc = Get-Process -Name "inkengine" -ErrorAction SilentlyContinue | Select-Object -First 1
        $effShow = if ($h.effort) { $h.effort } else { "unknown" }
        $visShow = if ($h.vision -eq $true) { "ON (0 MiB idle)" } else { "OFF" }
        Write-Host "[InkEngine] Status: RUNNING | PID=$($proc.Id) | Endpoint=$BaseUrl/v1 | Effort=$effShow | Vision=$visShow" -ForegroundColor Green
        Write-Host ("  Context : {0} / {1} tokens" -f $st.kv_cur_pos, $st.kv_max_ctx)
        Write-Host ("  VRAM    : Experts={0:N1} MiB | KV={1:N1} MiB (SWA {2:N1} + Global {3:N1} + Quest {4:N1})" -f `
            $st.expert_vram_mb, ($st.kv_swa_vram_mb + $st.kv_global_vram_mb + $st.kv_quest_vram_mb), `
            $st.kv_swa_vram_mb, $st.kv_global_vram_mb, $st.kv_quest_vram_mb)
        Write-Host ("  RAM     : Experts={0:N1} MiB | KV={1:N1} MiB" -f $st.expert_ram_mb, $st.kv_ram_mb)
        Write-Host ("  SSD     : KV Swap={0:N1} MiB (reads={1}, writes={2}) | Expert SSD reads={3}" -f `
            $st.kv_ssd_mb, $st.kv_ssd_reads, $st.kv_ssd_writes, $st.ssd_expert_reads)
        Write-Host ("  Cache   : VRAM hit={0:N1}% (Core={1:N1}%, Outer={2:N1}%) | Swaps={3}" -f `
            $st.vram_hit_pct, $st.core_hit_pct, $st.outer_hit_pct, $st.dma_promotions)
        exit 0
    }
    "restart" {
        Stop-InkServer
        Start-InkServerBackground -TargetCtx $Ctx -TargetEffort $Effort -TargetRamKv $RamKvMb
        exit 0
    }
    "start" {
        $h = Get-InkHealth
        if ($h -and (-not $CtxExplicit -or $h.max_ctx -eq $Ctx) -and -not $EffortExplicit) {
            Write-Host "[InkEngine] Already running in background (ctx=$($h.max_ctx))." -ForegroundColor Green
            exit 0
        }
        if ($h) { Stop-InkServer }
        Start-InkServerBackground -TargetCtx $Ctx -TargetEffort $Effort -TargetRamKv $RamKvMb
        exit 0
    }
}

# Default action: ensure background server is ready, then launch ONLY OpenCode CLI!
$health = Get-InkHealth
if ($health) {
    if (($CtxExplicit -and $health.max_ctx -ne $Ctx) -or $EffortExplicit) {
        Write-Host "[InkEngine] Restarting background server with requested settings (ctx=$Ctx, effort=$Effort)..." -ForegroundColor Yellow
        Stop-InkServer
        Start-InkServerBackground -TargetCtx $Ctx -TargetEffort $Effort -TargetRamKv $RamKvMb
    } else {
        Write-Host "[InkEngine] Background server already warm (ctx=$($health.max_ctx)) -> launching OpenCode CLI..." -ForegroundColor Green
    }
} else {
    Start-InkServerBackground -TargetCtx $Ctx -TargetEffort $Effort -TargetRamKv $RamKvMb
}

# Launch OpenCode CLI directly with inkengine/Inkling-2-Mini pre-selected!
$opencodeArgs = New-Object System.Collections.Generic.List[string]
$hasModel = $false
for ($j = 0; $j -lt $PassArgs.Count; $j++) {
    if ($PassArgs[$j] -eq "-m" -or $PassArgs[$j] -eq "--model") { $hasModel = $true }
}
if (-not $hasModel) {
    $opencodeArgs.Add("-m")
    $opencodeArgs.Add($ModelSpec)
}
foreach ($pa in $PassArgs) { $opencodeArgs.Add($pa) }

& opencode @opencodeArgs
exit $LASTEXITCODE
