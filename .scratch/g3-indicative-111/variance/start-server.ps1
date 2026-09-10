param(
    [Parameter(Mandatory=$true)][string]$Label
)
$base = "F:\ai\opencode\inference\.scratch\g3-indicative-111\variance"
$log = Join-Path $base "server-$Label.log"
$errLog = "$log.err"
$p = Start-Process -FilePath "F:\ai\opencode\inference\target\x86_64-pc-windows-msvc\release\ignis-server.exe" `
    -ArgumentList "--artifact","F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer" `
    -RedirectStandardOutput $log -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
$p.Id | Out-File -FilePath (Join-Path $base "server-$Label.pid") -Encoding ascii
Write-Output "started pid $($p.Id) for label $Label"
