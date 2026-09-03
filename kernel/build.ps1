# Build the ignis kernel leaf (CMake + Ninja + nvcc, SM120a).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File kernel/build.ps1
#
# Plain PowerShell, no developer prompt needed: imports the MSVC env itself.
# Pattern: ninfer's configure-ninja.ps1 / build-ninja.ps1 (proven on this
# machine, see NINFER_WINDOWS_BUILD_NOTES.md sections 10 and 12-14).

$ErrorActionPreference = "Stop"
$Kernel = Split-Path -Parent $MyInvocation.MyCommand.Path
# Optional -BuildDir override (default: the canonical kernel/build, which
# crates/*/build.rs link). A second build dir lets a parallel workstream
# verify new .cu files without contending on the canonical build.
$BuildDir = if ($args -and $args[0] -and ($args[0] -notlike '-*')) { $args[0] } else { "build" }
$BuildPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Kernel $BuildDir }

function Import-Vcvars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { Write-Error "vswhere not found at $vswhere"; exit 2 }
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $install) { Write-Error "no VS instance with C++ tools found"; exit 2 }
    $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
    foreach ($line in (cmd /c "`"$vcvars`" >nul && set")) {
        if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
    }
    Write-Host "[ignis-kernel] imported MSVC env from $vcvars"
}

Import-Vcvars

# Ninja (standalone install, NINFER_WINDOWS_BUILD_NOTES section 10).
if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    $fallbackDir = "F:\ai\q38\tools\ninja"
    if (Test-Path (Join-Path $fallbackDir "ninja.exe")) {
        $env:PATH = "$fallbackDir;$($env:PATH)"
        Write-Host "[ignis-kernel] ninja: fallback ($fallbackDir)"
    } else {
        Write-Error "ninja not found in PATH or $fallbackDir"; exit 2
    }
}

$CudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" }
# Forward slashes in the nvcc path: CMake (3.28) mis-quotes the generated
# CMakeCUDACompiler.cmake when the value contains backslashes + spaces.
# (The ninfer configure scripts use the same forward-slash form.)
$Nvcc = ("{0}/bin/nvcc.exe" -f ($CudaPath -replace '\\', '/'))
if (-not (Test-Path $Nvcc)) { Write-Error "nvcc not found at $Nvcc (set CUDA_PATH)"; exit 2 }

# NOTE: arguments are passed as an array — PowerShell backtick line
# continuations swallow leading whitespace and merge arguments into one
# string (the ninja scripts use the same array pattern).
$cmakeArgs = @(
    '-S', $Kernel,
    '-B', $BuildPath,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CUDA_ARCHITECTURES=120a',
    "-DCMAKE_CUDA_COMPILER=$Nvcc",
    '-DCMAKE_C_COMPILER_LAUNCHER=',
    '-DCMAKE_CXX_COMPILER_LAUNCHER=',
    '-DCMAKE_CUDA_COMPILER_LAUNCHER=',
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded'
)
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --build $BuildPath --parallel
exit $LASTEXITCODE