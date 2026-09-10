param(
    [Parameter(Mandatory=$true)][string]$OutCsv,
    [Parameter(Mandatory=$true)][int]$ServerPid,
    [int]$IntervalMs = 500
)
$ErrorActionPreference = "Continue"
"timestamp_ms,sm_clock_mhz,mem_clock_mhz,pstate,temp_c,power_w,util_gpu_pct,util_mem_pct,throttle_reasons,proc_cpu_pct,proc_threads,proc_workingset_mb" | Out-File -FilePath $OutCsv -Encoding utf8
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$proc = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
$lastCpuTime = $null
$lastSampleTime = $null
$cpuCount = [Environment]::ProcessorCount
while ($true) {
    $gpu = & nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks_event_reasons.active --format=csv,noheader,nounits 2>$null
    if (-not $gpu) { break }
    $parts = $gpu -split ','
    $now = Get-Date
    $p = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
    if (-not $p) { break }
    $cpuPct = ""
    if ($lastCpuTime -ne $null) {
        $deltaCpu = ($p.TotalProcessorTime - $lastCpuTime).TotalMilliseconds
        $deltaWall = ($now - $lastSampleTime).TotalMilliseconds
        if ($deltaWall -gt 0) {
            $cpuPct = [math]::Round(100.0 * $deltaCpu / $deltaWall / $cpuCount, 2)
        }
    }
    $lastCpuTime = $p.TotalProcessorTime
    $lastSampleTime = $now
    "$($sw.ElapsedMilliseconds),$($parts[0].Trim()),$($parts[1].Trim()),$($parts[2].Trim()),$($parts[3].Trim()),$($parts[4].Trim()),$($parts[5].Trim()),$($parts[6].Trim()),$($parts[7].Trim()),$cpuPct,$($p.Threads.Count),$([math]::Round($p.WorkingSet64/1MB,1))" | Out-File -FilePath $OutCsv -Append -Encoding utf8
    Start-Sleep -Milliseconds $IntervalMs
}
