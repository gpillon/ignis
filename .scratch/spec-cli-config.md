## Problem Statement

`ignis-server`'s only configuration surface is environment variables (`IGNIS_MODEL`, `IGNIS_BIND`, `IGNIS_ARTIFACT`, `IGNIS_TELEMETRY`, `IGNIS_ENABLE_THINKING`, `IGNIS_REASONING_EFFORT`). For a one-off run (manual testing, a quick GPU-profile invocation, an agent launching the server as a subprocess) the owner has to `set`/export a variable, run, then remember to unset it — there's no way to pass a value inline on the command that starts the binary, and no `--help`/`--version` to discover the surface without reading `README.md`.

## Solution

`ignis-server` accepts CLI flags that mirror the existing env-var config surface one-to-one, plus `--help`/`-h` and `--version`/`-V`. A flag overrides its env var, which overrides the built-in default — the same precedence already implicit in the env-var-only path (default → env), extended with a flag tier on top. Nothing about the env-var path changes: an invocation with no flags behaves exactly as today.

## User Stories

1. As the owner, I want to pass `--bind` on the command line, so that I can point a one-off server at a different port without exporting `IGNIS_BIND` first.
2. As the owner, I want to pass `--artifact <path>` on the command line, so that I can swap the loaded `.ninfer` container for a single run without touching my shell environment.
3. As the owner, I want to pass `--model` on the command line, so that I can override the reported model id for a quick test.
4. As the owner, I want to pass `--telemetry <path>` on the command line, so that I can redirect one run's JSONL sink without exporting `IGNIS_TELEMETRY`.
5. As the owner, I want to pass `--enable-thinking <true|false>` on the command line, so that I can flip the server-wide thinking default for a single invocation.
6. As the owner, I want to pass `--reasoning-effort <value>` on the command line, so that I can test a specific effort default without an env var.
7. As the owner, I want short aliases (`-b`, `-m`, `-a`, `-t`) for the most-used flags, so that interactive invocations stay terse.
8. As the owner, I want `--help`/`-h` to print the full flag surface (name, alias, corresponding env var, default), so that I don't have to open `README.md` to remember a flag name.
9. As the owner, I want `--version`/`-V` to print the crate version, so that I can confirm which build I'm running.
10. As an agent (qwen-code orchestration) launching `ignis-server` as a subprocess, I want to configure it entirely through argv, so that I don't have to mutate the parent process's environment (which could leak into sibling subprocesses) just to set one option.
11. As the owner, I want a flag to win over the matching env var when both are set, so that a quick `--bind` override on an already-exported `IGNIS_BIND` shell does what it looks like it does, with no silent env-var precedence surprise.
12. As the owner, I want an unset flag and an unset env var to fall back to today's documented default, so that existing invocations (systemd units, scripts, muscle memory) keep working unchanged.
13. As the owner, I want an unrecognized flag or a flag missing its required value to print a usage error to stderr and exit non-zero before any loader/scheduler work starts, so that a typo fails fast instead of starting the server with a misparsed value.
14. As the owner, I want `--enable-thinking`/`--reasoning-effort` flag values to go through the exact same parser as their env vars (`thinking::parse_default_enable_thinking` / `parse_default_reasoning_effort`), so that an invalid value is rejected identically regardless of which channel it came from.
15. As the owner, I want the resolved config (whichever tier each value came from) to be computable in a unit test with no process environment or real filesystem, so that precedence and validation are covered by fast tests instead of only end-to-end runs.
16. As the owner, I want `gpu-profile.ps1` and other launch scripts free to keep using env vars unchanged, so that this feature is additive and doesn't force a migration of existing tooling.

## Implementation Decisions

- New module `crates/server/src/config.rs`, exported from `ignis_server::config`, following the existing hand-rolled-parsing style of `crates/bench/src/main.rs` (`std::env::args().skip(1)`, manual `match`) — no new dependency (no `clap`/`argh`/etc.), consistent with the rest of the workspace.
- A `Config` struct holding the fully-resolved values `ignis-server`'s `main` currently computes inline: `model: String`, `bind: String`, `artifact: Option<PathBuf>`, `telemetry: Option<PathBuf>`, `enable_thinking: bool`, `reasoning_effort: Option<ReasoningEffort>`.
- A single pure entry point:
  ```
  pub fn resolve(
      args: &[String],
      env: impl Fn(&str) -> Option<String>,
  ) -> Result<ConfigOutcome, ConfigError>
  ```
  where `ConfigOutcome` is `Config` or a `Help`/`Version` variant (so `--help`/`--version` short-circuit resolution without touching the process — no `std::process::exit`/`println!` inside `resolve` itself). `env` is injected (not `std::env::var` called directly) so tests never touch real process environment. This is the one seam for this feature: `main` calls `config::resolve(&std::env::args().skip(1).collect::<Vec<_>>(), |k| std::env::var(k).ok())` and does nothing else config-related — every precedence/parsing/validation rule lives in `resolve`, testable without spawning a process.
- Precedence per field, evaluated independently (mixing tiers across fields is allowed: e.g. `--bind` from a flag while `IGNIS_ARTIFACT` still comes from the env is expected): CLI flag present → use it; else matching env var present → use it (existing `env()` helper's job today); else built-in default (`DEFAULT_MODEL`, `DEFAULT_BIND`, empty artifact/telemetry, thinking defaults `"true"`/`""`).
- Flag surface (long form is authoritative; short aliases are sugar for the same field):
  | Flag | Alias | Env var | Value |
  |---|---|---|---|
  | `--model <id>` | `-m` | `IGNIS_MODEL` | string |
  | `--bind <addr>` | `-b` | `IGNIS_BIND` | string |
  | `--artifact <path>` | `-a` | `IGNIS_ARTIFACT` | path |
  | `--telemetry <path>` | `-t` | `IGNIS_TELEMETRY` | path |
  | `--enable-thinking <true/false>` | — | `IGNIS_ENABLE_THINKING` | bool via `thinking::parse_default_enable_thinking` |
  | `--reasoning-effort <value>` | — | `IGNIS_REASONING_EFFORT` | `Option<ReasoningEffort>` via `thinking::parse_default_reasoning_effort` |
  | `--help` | `-h` | — | prints flag table + defaults, `ConfigOutcome::Help` |
  | `--version` | `-V` | — | prints `env!("CARGO_PKG_VERSION")`, `ConfigOutcome::Version` |

  No short aliases for `--enable-thinking`/`--reasoning-effort` (long, infrequent, a single-letter alias would just add confusion for little gain — kept out per the agreed scope of "the six flags plus help/version/short aliases", not an open-ended flag surface).
- Value parsing for `--enable-thinking`/`--reasoning-effort` reuses the existing `thinking::parse_default_enable_thinking`/`parse_default_reasoning_effort` functions verbatim — a flag's raw string goes through the identical validation as the env var's raw string does today, so behavior (including error messages) is channel-agnostic.
- `ConfigError` covers: unknown flag, flag missing its required value (e.g. `--bind` as the last arg), and a thinking-parse failure surfaced from the reused parsers — each carries a message suitable for `eprintln!("ignis-server: {err}"); std::process::exit(1);`, matching the refuse-to-start style already used for the sidecar/checksum/EOS failures in `main.rs`.
- `main.rs` changes: replace the current inline `env("IGNIS_MODEL", DEFAULT_MODEL)` / `env("IGNIS_BIND", DEFAULT_BIND)` / `env("IGNIS_ARTIFACT", "")` calls and the two `thinking::parse_default_*` calls with one `config::resolve(...)` call at the top of `main`, matched on `ConfigOutcome::{Config, Help, Version}`; `Help`/`Version` print and `std::process::exit(0)` before any loader/scheduler/telemetry-sink work runs. The `IGNIS_TELEMETRY` → `FileSink`/`StdoutSink` branching stays exactly where it is, just reading `config.telemetry` instead of calling `env("IGNIS_TELEMETRY", "")` inline.
- The private `env()` helper in `main.rs` either moves into `config.rs` or is deleted if `resolve`'s injected `env` closure replaces its call sites entirely — implementer's call, not load-bearing.
- No config file support (`--config <file>`), no new environment variables, no change to what's configurable — this is purely a second input channel for the six values that already exist, plus discoverability (`--help`/`--version`).
- `README.md`'s existing env-var table (around lines 215-218) gains the corresponding flag/alias per row rather than a separate new table, so the two channels are documented side by side.

## Testing Decisions

- Tests target `config::resolve` directly (unit tests in `crates/server/src/config.rs` or a sibling `#[cfg(test)]` module), passing synthetic `args: &[String]` and a closure/`HashMap`-backed `env` — no real process environment, no real filesystem, no server startup. This mirrors how `thinking.rs`'s `parse_default_enable_thinking`/`parse_default_reasoning_effort` and `loader.rs`'s functions are already unit-tested as pure functions taking inputs and returning `Result`.
- Cases to cover: no args/no env → all defaults; env only (today's existing behavior, must be unchanged) → env values win over defaults; flag only → flag values win over defaults; flag + matching env both set → flag wins, per field independently; unknown flag → `ConfigError`; flag missing its value → `ConfigError`; invalid `--enable-thinking`/`--reasoning-effort` value → the same error the env-var path already produces for that invalid value (assert message parity, not just "is an error"); `--help`/`--version` → `ConfigOutcome::Help`/`Version`, no error, and no other flags need to be valid alongside them (i.e. `ignis-server --help --nonsense` still just prints help — `--help` short-circuits before further parsing).
- No new integration/GPU test is needed: this doesn't touch the request path, the scheduler, or anything gated by ADR 0007's performance gate — it's argv/env resolution before any of that exists. Existing `main.rs` remains a thin, untested wire-up (consistent with today).

## Out of Scope

- A `--config <file>` / TOML/YAML config file.
- Any brand-new configurable value that doesn't already have an `IGNIS_*` env var today.
- Changing `IGNIS_GPU_PROFILE` (test-profile-only, not a server config value — untouched).
- Changing `ignis-bench`'s already-existing subcommand CLI (`replay`/`canary`/`report`/`gate`/`record`/`oracle`) — this spec is `ignis-server` only.
- Shell completions, man pages, or any packaging beyond `--help` text.

## Further Notes

- The two-tier precedence (env > default) generalizes to three tiers (flag > env > default) without breaking it — an env-var-only invocation is just "no flags supplied," so this is additive by construction and needs no migration of `gpu-profile.ps1` or any other existing launcher.
- `--help`'s printed table doubles as the up-to-date source for the `README.md` table once both exist; keeping them worded consistently (same default values, same env var names) avoids the two drifting apart.
