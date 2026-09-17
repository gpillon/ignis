param([string]$ProcName, [int]$IntervalSec = 2, [int]$Seconds = 60, [string]$Out)
$end = (Get-Date).AddSeconds($Seconds)
if (-not (Test-Path $Out)) { "t,adapter_ded_mib,adapter_shared_mib,proc_pid,proc_ded_mib,proc_shared_mib,proc_commit_mib,proc_ws_mib,smi_used_mib" | Out-File -Encoding utf8 $Out }
while ((Get-Date) -lt $end) {
  $t = Get-Date -Format 'HH:mm:ss'
  $ad = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
  $aded = [math]::Round((($ad | ? { $_.Path -like '*dedicated*' } | Measure-Object CookedValue -Maximum).Maximum)/1MB)
  $ash = [math]::Round((($ad | ? { $_.Path -like '*shared*' } | Measure-Object CookedValue -Maximum).Maximum)/1MB)
  $p = Get-Process -Name $ProcName -ErrorAction SilentlyContinue | Select-Object -First 1
  $pded=''; $psh=''; $pcm=''; $pws=''; $ppid=''
  if ($p) {
    $ppid = $p.Id; $pws = [math]::Round($p.WorkingSet64/1MB)
    $s = (Get-Counter "\GPU Process Memory(pid_$($p.Id)_*)\Dedicated Usage","\GPU Process Memory(pid_$($p.Id)_*)\Shared Usage","\GPU Process Memory(pid_$($p.Id)_*)\Total Committed" -ErrorAction SilentlyContinue).CounterSamples
    $pded = [math]::Round((($s | ? { $_.Path -like '*dedicated*' } | Measure-Object CookedValue -Sum).Sum)/1MB)
    $psh = [math]::Round((($s | ? { $_.Path -like '*shared usage*' } | Measure-Object CookedValue -Sum).Sum)/1MB)
    $pcm = [math]::Round((($s | ? { $_.Path -like '*committed*' } | Measure-Object CookedValue -Sum).Sum)/1MB)
  }
  $smi = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()
  "$t,$aded,$ash,$ppid,$pded,$psh,$pcm,$pws,$smi" | Out-File -Append -Encoding utf8 $Out
  Start-Sleep -Seconds $IntervalSec
}
