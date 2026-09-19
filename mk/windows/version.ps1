# The release version, for the Makefile (the Windows VERSION_TOOL hook,
# mk/os/windows.mk). The counterpart of mk/linux/version.sh -- same three
# actions, same files, same output.
#
#   show           print the version each file declares
#   check          exit 1 when they disagree: the comparison
#                  .github/workflows/release.yml makes before it builds
#                  anything (Cargo.toml against web/package.json), runnable
#                  before you tag -- plus the two lockfiles, which the
#                  workflow does not look at and the next build would rewrite
#   set <version>  write an explicit version, `x.y.z` with an optional
#                  `-prerelease`
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

# `x.y.z`, optionally `-prerelease`. Semver's `+build` is deliberately not
# accepted: `npm version` drops build metadata, so it would land in the two
# Rust files and not in the two web ones -- and the release workflow compares
# those two by string, so such a version could never tag at all.
$VersionPattern = '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'

function Die([string]$Message) {
    [Console]::Error.WriteLine("error: $Message")
    exit 1
}

# Put back what a failed step already wrote, and say so either way: a rollback
# that quietly failed would leave a half-bumped tree behind a message claiming
# it did not.
function Restore-Files([string[]]$Paths) {
    if (Invoke-Tool 'git' (@('checkout', '--') + $Paths)) {
        [Console]::Error.WriteLine("restored: $($Paths -join ' ')")
    } else {
        [Console]::Error.WriteLine("warning: could not restore $($Paths -join ' ') -- check them by hand")
    }
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

# What the lockfile says the workspace members are at: every `ignis-*`
# entry's version, deduplicated. One value when they agree -- which is what
# `check` wants to compare -- and a `/`-joined list when they do not, so a
# single stale member is visible rather than hidden behind the first one.
function Get-CargoLockVersion {
    $found = [regex]::Matches((Read-File 'Cargo.lock'), '(?m)^name = "ignis-[a-z-]*"\r?\nversion = "(.*)"') |
        ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
    return ($found -join '/')
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

# Say so when those files already carry changes, and carry on. It was a
# refusal at first, and that was wrong twice over: a second bump in the same
# session is ordinary, and repairing a tree a half-done bump left is the case
# this tool exists for. What the refusal was really guarding -- a release
# commit sweeping up unrelated work -- is handled where it belongs, by the
# `git commit` line printed below naming its four files.
function Warn-IfDirty {
    $dirty = & git status --porcelain -- Cargo.toml Cargo.lock web/package.json web/package-lock.json 2>$null
    if ($dirty) {
        [Console]::Error.WriteLine("note: the version files already carry changes:`n" + ($dirty -join "`n"))
    }
}

function Set-Version([string]$New) {
    $old = Get-CargoVersion
    if (-not $old) { Die 'no version in Cargo.toml [workspace.package]' }
    # Refuse only when every file already says it. Setting the version a
    # disagreeing tree half-carries is the repair this exists for -- that is
    # the state a half-done bump leaves, and what the failed v0.1.1 tag was.
    if ($New -eq $old -and $New -eq (Get-CargoLockVersion) -and
        $New -eq (Get-JsonVersion 'web/package.json') -and
        $New -eq (Get-JsonVersion 'web/package-lock.json')) {
        Die "every file already declares $New"
    }

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
        Restore-Files @('Cargo.toml')
        Die 'cargo could not refresh Cargo.lock'
    }

    # npm owns both web files: package.json and the two places the lockfile
    # repeats the version. --no-git-tag-version keeps it out of git entirely.
    if (-not (Invoke-Tool 'npm' @('version', $New, '--no-git-tag-version', '--allow-same-version') 'web')) {
        Restore-Files @('Cargo.toml', 'Cargo.lock')
        Die 'npm could not set the Playground version'
    }

    # Never report a bump the release workflow would refuse: the two tools
    # above each normalize what they are given (npm drops build metadata,
    # for one), so what they wrote is checked rather than assumed.
    $written = @((Get-CargoVersion), (Get-CargoLockVersion),
                 (Get-JsonVersion 'web/package.json'), (Get-JsonVersion 'web/package-lock.json'))
    if ($written | Where-Object { $_ -ne $New }) {
        Show-Versions
        Die "the files did not all take $New -- fix them before tagging"
    }

    if ($old -eq $New) { Write-Host "repaired $New" } else { Write-Host "$old -> $New" }
    Show-Versions
    Write-Host ''
    Write-Host 'next:'
    Write-Host "  git commit -m 'ignis $New' -- Cargo.toml Cargo.lock web/package.json web/package-lock.json && git push origin main"
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
            Die "'$Value' is not a version: expected x.y.z, optionally -prerelease (e.g. 1.2.3, 1.2.3-rc.1). Semver +build metadata is not accepted -- npm drops it, and the release workflow compares Cargo.toml against web/package.json by string"
        }
        Warn-IfDirty
        Set-Version $Value
    }
    'bump' {
        if (-not $Value) { Die 'bump needs patch, minor or major' }
        Warn-IfDirty
        Step-Version $Value
    }
}
