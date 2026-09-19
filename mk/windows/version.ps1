# The release version, for the Makefile (the Windows VERSION_TOOL hook,
# mk/os/windows.mk). The counterpart of mk/linux/version.sh -- same three
# actions, same files, same output.
#
#   show           print the version each file declares
#   check          exit 1 when they disagree (what .github/workflows/release.yml
#                  checks before it builds anything, runnable before you tag)
#   set <version>  write an explicit version, `x.y.z` with an optional
#                  `-prerelease` and `+build`
#   bump <part>    patch, minor or major, from what the workspace declares
#
# The version lives in four files and every one of them must agree: the
# release workflow refuses a tag whose Cargo.toml and web/package.json differ
# (the Playground ships inside the binary, so one version covers both), and a
# lockfile left behind makes the next build rewrite it under you.
#
#   Cargo.toml              [workspace.package] version -- the source of truth
#   Cargo.lock              every workspace member's entry
#   web/package.json        the Playground's own version
#   web/package-lock.json   its root and its "" package entry
#
# Only the first of those is edited here. `cargo update --workspace` owns the
# Cargo lockfile and `npm version` owns both web files -- hand-written JSON
# surgery on a lockfile is how every nested dependency ends up carrying the
# workspace's version number.
#
# Nothing here commits or tags: it edits the files and prints the commands,
# because a tag push builds and publishes a release image and that is the
# operator's call, not a side effect of a version bump.
#
# The one write goes through [IO.File]::WriteAllText with UTF-8 and no BOM,
# and keeps the file's own newlines: PowerShell's own redirection would hand
# git a whole-file diff, or a BOM that breaks `sed -n` on the other side.

param(
    [Parameter(Position = 0)]
    [ValidateSet('show', 'check', 'set', 'bump')]
    [string]$Action = 'show',
    [Parameter(Position = 1)]
    [string]$Value
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Repo

# `x.y.z`, optionally `-prerelease` and `+build` (semver 2.0.0's grammar,
# minus the leading-zero rule -- `01.0.0` is nobody's typo worth a rejection
# message).
$VersionPattern = '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'

function Die([string]$Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

# Run a native tool quietly and report success. `Continue` for the duration:
# see the note at the cargo call below for why `Stop` cannot see this through.
function Invoke-Tool([string]$Exe, [string[]]$Arguments, [string]$WorkingDirectory) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $here = Get-Location
    try {
        if ($WorkingDirectory) { Set-Location (Join-Path $Repo $WorkingDirectory) }
        & $Exe @Arguments 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } finally {
        Set-Location $here
        $ErrorActionPreference = $previous
    }
}

function Read-File([string]$Path) {
    return [IO.File]::ReadAllText((Join-Path $Repo $Path))
}

function Write-File([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText((Join-Path $Repo $Path), $Text, (New-Object Text.UTF8Encoding $false))
}

function Get-CargoVersion {
    $section = [regex]::Match((Read-File 'Cargo.toml'), '(?ms)^\[workspace\.package\]\r?\n(.*?)(?=^\[)')
    if (-not $section.Success) { return '' }
    $m = [regex]::Match($section.Groups[1].Value, '(?m)^version *= *"(.*)"')
    if ($m.Success) { return $m.Groups[1].Value } else { return '' }
}

# The lockfile's entry for one workspace member -- the line after its name.
function Get-CargoLockVersion {
    $m = [regex]::Match((Read-File 'Cargo.lock'), '(?m)^name = "ignis-server"\r?\nversion = "(.*)"')
    if ($m.Success) { return $m.Groups[1].Value } else { return '' }
}

function Get-JsonVersion([string]$Path) {
    $m = [regex]::Match((Read-File $Path), '(?m)^  "version": "(.*)",$')
    if ($m.Success) { return $m.Groups[1].Value } else { return '' }
}

function Show-Versions {
    Write-Host ("Cargo.toml            {0}" -f (Get-CargoVersion))
    Write-Host ("Cargo.lock            {0}" -f (Get-CargoLockVersion))
    Write-Host ("web/package.json      {0}" -f (Get-JsonVersion 'web/package.json'))
    Write-Host ("web/package-lock.json {0}" -f (Get-JsonVersion 'web/package-lock.json'))
}

function Test-Versions {
    $cargo = Get-CargoVersion
    if (-not $cargo) { Die 'no version in Cargo.toml [workspace.package]' }
    $lock = Get-CargoLockVersion
    $web = Get-JsonVersion 'web/package.json'
    $webLock = Get-JsonVersion 'web/package-lock.json'
    Show-Versions
    $failed = $false
    if ($cargo -ne $web) {
        Write-Host "error: Cargo.toml ($cargo) and web/package.json ($web) disagree -- the release workflow refuses this" -ForegroundColor Red
        $failed = $true
    }
    if ($cargo -ne $lock) {
        Write-Host "error: Cargo.lock ($lock) is stale -- run 'cargo update --workspace --offline'" -ForegroundColor Red
        $failed = $true
    }
    if ($cargo -ne $webLock) {
        Write-Host "error: web/package-lock.json ($webLock) is stale" -ForegroundColor Red
        $failed = $true
    }
    if ($failed) { exit 1 }
    Write-Host "ok: every file declares $cargo"
}

# Refuse to edit a file that already carries changes: a bump is a mechanical
# rewrite, and mixing it into unrelated work is how a release commit ends up
# carrying something nobody reviewed.
function Assert-Clean {
    $dirty = & git status --porcelain -- Cargo.toml Cargo.lock web/package.json web/package-lock.json 2>$null
    if ($dirty) { Die ("the version files already have uncommitted changes:`n" + ($dirty -join "`n")) }
}

function Set-Version([string]$New) {
    $old = Get-CargoVersion
    if (-not $old) { Die 'no version in Cargo.toml [workspace.package]' }
    if ($New -eq $old) { Die "the workspace already declares $New" }

    # Only inside [workspace.package]: `version = "1"` appears under a dozen
    # dependencies in the same file.
    $toml = Read-File 'Cargo.toml'
    $toml = [regex]::Replace($toml, '(?ms)(^\[workspace\.package\]\r?\n(?:(?!^\[).*?)^version *= *")[^"]*(")',
        { param($m) $m.Groups[1].Value + $New + $m.Groups[2].Value }, 1)
    Write-File 'Cargo.toml' $toml

    # Cargo owns its lockfile: --offline touches nothing but the workspace
    # members' own entries, and never reaches the network. Native stderr is
    # captured rather than redirected: under Windows PowerShell a redirected
    # native stderr line arrives as an ErrorRecord, which `Stop` turns into a
    # failure even when the exe returned 0 (cargo prints "Locking N packages"
    # there).
    if (-not (Invoke-Tool 'cargo' @('update', '--workspace', '--offline'))) {
        & git checkout -- Cargo.toml
        Die "cargo could not refresh Cargo.lock (Cargo.toml left at $old)"
    }

    # npm owns both web files: package.json and the two places the lockfile
    # repeats the version. --no-git-tag-version keeps it out of git entirely.
    if (-not (Invoke-Tool 'npm' @('version', $New, '--no-git-tag-version', '--allow-same-version') 'web')) {
        & git checkout -- Cargo.toml Cargo.lock
        Die 'npm could not set the Playground version (nothing changed)'
    }

    Write-Host "$old -> $New"
    Show-Versions
    Write-Host ''
    Write-Host 'next:'
    Write-Host "  git commit -am 'ignis $New' && git push origin main"
    Write-Host "  git tag -a v$New -m 'ignis $New' && git push origin v$New   # builds and publishes the release"
}

function Step-Version([string]$Part) {
    $current = Get-CargoVersion
    if ($current -notmatch $VersionPattern) {
        Die "the workspace version '$current' is not x.y.z -- use V=<version>"
    }
    $core = ($current -split '[-+]')[0]
    $pre = if ($current -match '-') { ($current -split '-', 2)[1] } else { '' }
    $parts = $core -split '\.'
    [int]$major = $parts[0]; [int]$minor = $parts[1]; [int]$patch = $parts[2]
    switch ($Part) {
        # A prerelease bumps to its own release, the way `npm version patch`
        # does: 1.2.3-rc.1 patches to 1.2.3, not 1.2.4.
        'patch' { if (-not $pre) { $patch++ } }
        'minor' { $minor++; $patch = 0 }
        'major' { $major++; $minor = 0; $patch = 0 }
        default { Die "unknown part '$Part' (expected patch, minor or major)" }
    }
    Set-Version "$major.$minor.$patch"
}

switch ($Action) {
    'show' { Show-Versions }
    'check' { Test-Versions }
    'set' {
        if (-not $Value) { Die 'set needs a version' }
        if ($Value -notmatch $VersionPattern) {
            Die "'$Value' is not a version: expected x.y.z, optionally -prerelease and +build (e.g. 1.2.3, 1.2.3-rc.1, 1.2.3+cuda13)"
        }
        Assert-Clean
        Set-Version $Value
    }
    'bump' {
        if (-not $Value) { Die 'bump needs patch, minor or major' }
        Assert-Clean
        Step-Version $Value
    }
}
