# Copy, verify and patch-record the kernel leaf's vendored reference subtree
# (ADR 0010). See kernel/vendor/VENDOR.md.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/vendor-ninfer.ps1 verify
#   ... vendor-ninfer.ps1 sync --reference F:/ai/q38/ninfer
#   ... vendor-ninfer.ps1 record-patch src/core/arena.cu --reason "why"
#
# A thin wrapper over the `vendor-ninfer` binary of crates/vendor, so the logic
# is covered by `cargo test` instead of living in an untested script. Every
# argument after the command is forwarded verbatim.
#
# Exit codes (from the binary): 0 clean, 1 a verification finding, 2 a usage or
# I/O error.

$ErrorActionPreference = "Stop"
$Repository = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (-not (Get-Command cargo.exe -ErrorAction SilentlyContinue)) {
    Write-Error "cargo not found in PATH (the vendoring tool is crates/vendor)"; exit 2
}

Push-Location $Repository
try {
    # --quiet: cargo's build chatter would bury the verification report.
    & cargo run --quiet --package ignis-vendor --bin vendor-ninfer -- @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
