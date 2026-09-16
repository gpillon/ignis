# Build the one-off chat-render fixture recorder (GitHub #184) against the
# reference's existing `build-ninja` tree: the same static libraries, include
# directories and compile flags as `tools/vision-fixtures/build.ps1`.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/chat-render-fixtures/build.ps1
#
# Produces `$OutDir\record.exe`. Nothing here touches the GPU.
param(
    [string]$Ninfer = 'F:\ai\q38\ninfer',
    [string]$OutDir = (Join-Path $env:TEMP 'ignis-chat-render-recorder'),
    [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    [string]$Cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
)
$ErrorActionPreference = 'Stop'

$Build = Join-Path $Ninfer 'build-ninja'
$Vcpkg = Join-Path $Build 'vcpkg_installed\x64-windows'
$Source = Join-Path $PSScriptRoot 'record.cpp'
New-Item -ItemType Directory -Force $OutDir | Out-Null

$includes = @('include', 'src', 'third_party', 'third_party\utf8proc',
    'src\targets\qwen3_6\export', 'src\targets\qwen3_6\impl') |
    ForEach-Object { "/I`"$(Join-Path $Ninfer $_)`"" }
$libs = @('ninfer_engine', 'ninfer_core', 'ninfer_artifact', 'ninfer_ops', 'ninfer_nvfp4_tma',
    'ninfer_core', 'ninfer_text', 'ninfer_media_decode') |
    ForEach-Object { "`"$(Join-Path $Build "src\$_.lib")`"" }
$ffmpeg = @('avformat', 'avcodec', 'swresample', 'swscale', 'avutil') |
    ForEach-Object { "`"$(Join-Path $Vcpkg "lib\$_.lib")`"" }
$system = 'cudart_static.lib secur32.lib ncrypt.lib crypt32.lib kernel32.lib user32.lib advapi32.lib'

$compile = "cl /nologo /EHsc /O2 /DNDEBUG /MD /std:c++20 /Zc:preprocessor /utf-8 " +
    "/DNOMINMAX /DWIN32_LEAN_AND_MEAN /DUTF8PROC_STATIC " +
    "/I`"$Cuda\include`" $($includes -join ' ') `"$Source`" /Fo`"$OutDir\\`" /Fe`"$OutDir\record.exe`" " +
    "/link /NODEFAULTLIB:LIBCMT /LIBPATH:`"$Cuda\lib\x64`" $($libs -join ' ') $($ffmpeg -join ' ') $system"
cmd /c "call `"$VcVars`" >nul && $compile"
if ($LASTEXITCODE -ne 0) { throw "recorder build failed ($LASTEXITCODE)" }
Copy-Item (Join-Path $Vcpkg 'bin\*.dll') $OutDir -Force
Write-Host "built $OutDir\record.exe"
