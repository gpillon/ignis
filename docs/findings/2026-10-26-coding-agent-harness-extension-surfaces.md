# Coding-agent harness extension surfaces: one skill folder and one MCP server reach opencode, pi, DeepSeek Harness and qwen-code; their code plugins are three different APIs, and qwen-code has none

- Kind: research
- Status: current
- Observed: 2026-10-26
- Last verified: 2026-10-26
- Scope: connectors / plugin and extension APIs, MCP clients, Agent Skills, permissions, provider configuration and
  distribution of opencode v1.19.4, pi 1.4.0 (`main` `b84d5d1`), DeepSeek Harness 0.2.1-rc.3 and qwen-code 0.26.1
  (installed 0.23.0), for reaching `/v1/systemone`, `/v1/images/*`, `/v1/audio/transcriptions` and `PATCH /v1/config`
- Topic: harness-and-build
- Related: [#383](https://github.com/gpillon/ignis/issues/383), [System One decisions](../user/api/decide.md),
  [Connect a client](../user/start/connect-a-client.md), [the agent path](../user/paths/agent.md),
  [ADR 0071](../adr/0071-the-server-downloads-and-removes-models-from-the-server-view.md)
- Superseded by: none
- Summary: opencode, pi, DeepSeek Harness and qwen-code all read Agent Skills and are MCP clients, while three have code plugins on three different TypeScript APIs; the connector's design is open on #383.

## Question

ignis's owner wants connectors so that coding-agent harnesses (opencode, pi, DeepSeek Harness, qwen-code, which runs
the owner's own orchestration, maybe others later) can use what ignis serves beyond chat: System One decisions, image
generation and editing, transcription, and runtime configuration. Which mechanisms does each harness offer for that
(native plugin or extension, MCP client, Agent Skills, lighter tool files), and what does each one allow and cost?
This is the fact base for choosing between one shared mechanism and a plugin per harness; it does not make that
choice.

## Evidence

Primary sources only: each harness's official docs and its source at a named commit, the npm registry and the GitHub
API, read on 2026-10-26. Four research agents wrote one file per harness under the same nine headings, with every
claim cited to a doc page or a `path@commit`; the per-harness files, with the line-level citations this summary
abbreviates, are in `.scratch/connectors-research/` (`opencode.md`, `pi.md`, `deepseek-harness.md`, `qwen-code.md`;
untracked).

| Harness | Source read | Docs |
|---|---|---|
| opencode (MIT, TypeScript on Bun) | `github.com/anomalyco/opencode` tag `v1.19.4`, `148842ef` (2026-10-22); `sst/opencode` redirects there. Installed locally: 1.18.35, same plugin hook set | `opencode.ai/docs` (generated from `packages/web/src/content/docs/*.mdx` at the same tag) |
| pi (MIT, TypeScript) | `github.com/earendil-works/pi` `main` `b84d5d1` (2026-10-26), package `@earendil-works/pi-coding-agent`; latest npm release 1.4.0 (`v1.4.0` = `4a08025`) | `pi.dev`, `packages/coding-agent/docs/*.md` |
| DeepSeek Harness, `dsh` (MIT, TypeScript, developer preview) | `github.com/deepseek-ai/deepseek-harness` `master` `22ddee37` (2026-10-23), version 0.2.1-rc.3; npm `@deepseek-ai/dsh` `latest` 0.2.1-rc.2 | `deepseek.com/en/harness/`, `deepseek-harness.github.io/deepseek-harness/`, `docs/` in the repo |
| qwen-code (Apache-2.0, TypeScript on Node) | `github.com/QwenLM/qwen-code` tag `v0.26.1`, `b475278c` (2026-10-24, npm `latest`); installed locally: 0.23.0 (`v0.23.0` = `98a9c964`), on which the facts used below were re-checked (only the experimental Mods are absent there). "Originally based on Google Gemini CLI v0.8.2", independent since v0.1 (2025-10-27); not a GitHub fork | `qwenlm.github.io/qwen-code-docs` (built from `docs/`; the repo's `docs/` at the tag was read) |

Two cross-checks against ignis were made afterwards, by reading source on both sides:

- pi's System One client (`packages/ai/src/api/system-one-shared.ts@b84d5d1`, `parseAnswers`) against ignis's
  `crates/server/src/decide.rs` (`enum Answer`, `struct Usage`, `struct DecideRequest`).
- dsh's model layer (`packages/llm/llm-pi-ai/src/catalog.ts@22ddee37`) for whether pi-ai's classifier and image APIs
  reach dsh.

## Finding

Observed facts, harness by harness, on the axes a connector depends on:

| Axis | opencode | pi | DeepSeek Harness | qwen-code |
|---|---|---|---|---|
| **Native extension** | In-process JS/TS plugin, 21 typed hooks (`tool`, `tool.execute.before/after`, `chat.params`, `chat.headers`, `experimental.chat.system.transform`, …); tools typed with Zod. A second, undocumented "v2" plugin API is already exported by `@opencode-ai/plugin` | In-process TypeScript extension, about 41 events; tools typed with TypeBox; can register commands, providers (chat, image and classifier models), MCP servers, virtual models and TUI | Cordis plugin in an npm "bundle" with full internal API: tools, `tools/pre-execute` and other hooks, system-prompt sections, slash commands, agent presets, subagent providers, LLM adapters, browser UI slots | **Declarative only**: a `qwen-extension.json` bundle of MCP servers, LSP servers, context files, Markdown commands, skills, agents, workflows, hooks (22 events; can block, ask or rewrite tool input) and settings handed to MCP servers as env vars. No field for native tools, providers or the system prompt. Experimental Mods (not in 0.23.0) run in QuickJS with no network |
| **API stability** | `@opencode-ai/plugin` in lockstep with opencode; 33 releases in about 3 months; no breaking-change policy; `engines.opencode` gate for npm plugins | Breaking extension-API changes in 1.0.3 and 1.2.0 | README: "THERE WILL BE COMPATIBILITY-BREAKING CHANGES"; `docs/upgrade-guide/` holds 22 + 12 + 23 + 1 breaking-change guides for alpha.1, alpha.2, rc.2, rc.3; `engines.dsh` enforced | Stable release every 2–5 days; manifest has no schema version; documented surface churn (TOML commands deprecated, hook `timeout` unit changed) |
| **Lighter tool mechanism** | Tool files in `.opencode/tools/` or `~/.config/opencode/tools/` (same `tool()` helper, no hooks); Markdown slash commands with `` !`cmd` `` injection | None besides extensions and skills; the project's original stance was CLI tools described in READMEs | None besides bundles and skills; a config-only bundle can add an MCP server | `tools.discoveryCommand` + `tools.callCommand` (a program prints declarations, another runs a call with JSON on stdin); Markdown commands with `!{cmd}`; subagents in `.qwen/agents/` |
| **MCP client** | Yes: stdio, Streamable HTTP with SSE fallback; static headers and OAuth (DCR); prompts become slash commands, resources become three tools | Yes since 0.99.0 (2026-09-29): stdio and Streamable HTTP (SSE refused); headers and OAuth. Default `exposure: codemode`: tools are reached from pi's JavaScript sandbox, not declared to the model; `direct` and `deferred` exist | Yes: stdio and Streamable HTTP only; static headers only, no OAuth; no prompt templates; configured in YAML, no MCP settings page | Yes: stdio, Streamable HTTP (`httpUrl`), SSE (`url`); headers and OAuth; tools **deferred by default** behind `tool_search` unless `alwaysLoadTools: true`; project `.mcp.json` servers untrusted until approved; request timeout default 600 s |
| **MCP or tool image result → model** | Yes, only as a base64 `data:` URL, and with `@ai-sdk/openai-compatible` only via a synthetic user message that reaches the model when the model entry declares `modalities.input: ["image"]`; otherwise the model reads `ERROR: Cannot read image` | Yes (`ImageContent`); on an OpenAI-compatible endpoint the model entry must declare `input: ["text","image"]` | Yes, when the routed model declares image input and the attachment store is mounted (it is in the base bundle) | Yes: moved into a follow-up user message with `image_url` data URLs on the OpenAI wire, only when the model's image modality is on (`generationConfig.modalities`); an unknown model id defaults to text-only |
| **Tool image → user** | No renderer in the TUI at this tag; ACP editors get it | The terminal shows extension tool images | Proper card only for the built-in `read_image` or a plugin that ships its own client card; otherwise the image block is flattened JSON. Markdown `![](path)` in the assistant's text displays | The TUI shows it (native on Kitty/Ghostty, a `chafa` preview elsewhere, else a placeholder); headless, ACP, Web Shell and IDE surfaces do not |
| **Agent Skills (`SKILL.md`)** | `.opencode/skills`, `.claude/skills`, `.agents/skills` (project, walking up), `~/.config/opencode/skills`, `~/.claude/skills`, `~/.agents/skills`, `skills.paths`, remote `skills.urls`; every skill is also a slash command | `~/.pi/agent/skills`, `.pi/skills`, `~/.agents/skills`, project `.agents/skills` from cwd up to the repo root (project ones after trust) | `.dsh/skills`, `.agents/skills` (nearest `.git` ancestor), `~/.dsh/skills`, `~/.agents/skills`; one level deep only | `.qwen/skills`, `.agents/skills` **in the launch directory only** (no walk up), `~/.qwen/skills`, `~/.agents/skills`, `skills.directories`, extension skills; every skill is also `/<name>` |
| **Skill frontmatter honoured** | Source checks only `name`, `description` | `name`, `description`, `disable-model-invocation`; `allowed-tools` ignored | `name`, `description`, `whenToUse`, `metadata`, `disable-model-invocation`, `user-invocable`; camelCase legacy keys throw | `name`, `description`, `allowedTools` (camelCase: becomes session allow rules when the skill runs), `model`, `argument-hint`, `when_to_use`, `disable-model-invocation`, `user-invocable`, `paths`, `priority`, `hooks`; kebab `allowed-tools` ignored |
| **Skill scripts** | Yes, run through `bash` | Yes, run through `bash` | Yes, run through the shell tool | Yes, run through the shell tool |
| **Gate on `curl` to localhost** | None by default (`"*": "allow"`); MCP tools ask under their own name; plugin and file tools are not gated unless they call `ctx.ask()` | None: no permission system ("YOLO"); gating is an extension's `tool_call` handler; tools can only declare hints (`readOnlyHint`, `destructiveHint`) | None by default: the sandbox governs file effects only, "network stays unrestricted"; `ask` fires on sandbox escalation, on a tool that asks for itself, or from a hook | **Default approval mode `auto`**: a non-read-only shell command such as `curl` goes to a two-stage LLM classifier on `fastModel`, else the main model; a specific `Bash(curl …)` allow rule skips it, a `WebFetch(127.0.0.1)` allow does not; MCP tools go to the classifier unless `trust: true` or allowed by name |
| **`web_fetch` / `webfetch` POST** | No (`url` + `format` only) | — | No (`url` only) | — |
| **Second model for side jobs** | Title: `small_model`, else a same-provider `gemini-flash`/`gpt-nano`/`claude-haiku` family model, else the main model; compaction: the session model | None: compaction and summaries on the session model, no titles | Title and compaction inherit the current route; `web_search` calls DeepSeek's API and needs a DeepSeek key | Slots `fastModel`, `compactionModel`, `visionModel`, `imageModel`, `voiceModel`; side queries (web-fetch summaries, next-speaker check, recap, …) run on `fastModel`, else the main model; titles and tool-use summaries are skipped without `fastModel`; background memory agents (on by default) run on the main model |
| **Volatile prompt content** | System prompt: model id, cwd, workspace root, git yes/no, platform, **today's date** (day resolution); no switch; only `experimental.chat.system.transform` rewrites it | `cwd` only (the date was removed in 0.80.7) | None in the system prompt: time, cwd and `AGENTS.md` go into appended user-role messages, each with an off switch except cwd | Startup user message with date, OS, cwd and a folder tree (`model.skipStartupContext` drops them) and a git snapshot (`git status` + 5 commits; no switch on a local run); a date reminder on the first turn of a new day; the system prompt is rebuilt mid-session when auto-memory writes |
| **Distribution** | npm (`opencode plugin <module>`; a bare name means `@latest`), local files or paths; no official marketplace | "pi packages" on npm or git: `pi install`, `pi update`; gallery at `pi.dev/packages` | npm bundle via `dsh plugin add` or the Plugins page; git; tarball; opt-in JSON plugin index | `qwen extensions install` from git, `owner/repo`, local dir, archive, scoped npm package or a Claude marketplace; it also converts Gemini CLI extensions, Claude Code plugins and Qoder plugins; `qwen extensions update` |

What each harness already has for ignis's routes:

- **System One.** pi has a client: the `typesafe-system-one` classifier API POSTs `{baseUrl}/systemone` with `bool`
  sent as `noul`, plus `choice` and `score`. For text decisions it is compatible with ignis as both stand: pi's
  `parseAnswers` requires `noul` for a bool, `choice` + `probabilities` + `confidence`, `score` + `confidence`, and
  reads `usage`; ignis's `Answer` (`#[serde(tag = "type", rename_all = "lowercase")]`) and `Usage { input_tokens,
  output_tokens }` carry exactly those, and pi ignores ignis's extra `legend`. A classifier model at a new URL can be
  declared only by an extension (`pi.registerProvider` with `type: "classifier"`, `api: "typesafe-system-one"` and
  `typesafeSystemOneApi()`), as pi's built-in llama.cpp extension does for llama.cpp's `/v1/systemone`; `models.json`
  cannot. Pointing pi's built-in `typesafe` provider at ignis sends `model: "jev-latest"`, which ignis refuses with
  `422 model_not_found`. pi needs a non-empty API key for it. Classifier models are reached from codemode or
  `ctx.modelRegistry.classify()`, not from `/model`. dsh uses pi's model library (`@earendil-works/pi-ai@1.3.0`,
  patched) but its adapter dispatches chat only: `catalog.ts` states a provider with image-generation or classifier
  models alone "is not a catalog route", and nothing under `packages/llm/llm-pi-ai/src` calls `classify` or
  `generateImages`. opencode and qwen-code have no System One client; qwen-code's yes/no checks (the approval
  classifier, prompt hooks, the next-speaker check) all go through chat completions.
- **Images.** No harness has a client for `/v1/images/generations` or `/v1/images/edits`. pi's only built-in
  image-generation wire format is OpenRouter's chat-completions `modalities`. qwen-code has an `image_gen` tool, but it
  speaks DashScope's `/services/aigc/multimodal-generation/generation`, refuses any `baseUrl` that is not `https:`,
  and downloads the result from a public HTTPS URL.
- **Transcription.** No harness has a `/v1/audio/transcriptions` client (opencode was not checked for one). dsh's voice
  input is a local SenseVoice model. qwen-code's dictation sends an `input_audio` part to `/chat/completions` and only
  for `qwen3-asr-flash*` model ids (a code comment says DashScope does not serve `/audio/transcriptions`); it accepts
  a loopback `http` base URL.
- **Configuration.** No harness has anything like `PATCH /v1/config`.

Pointing each harness at ignis for chat works through its OpenAI-compatible provider: opencode's `provider` entry with
`npm: "@ai-sdk/openai-compatible"`, pi's `~/.pi/agent/models.json`, dsh's "Custom model API" (`openai-completions`,
`openai-responses` or `anthropic-messages`), qwen-code's `modelProviders.openai[]` (`id`, `baseUrl`, `envKey`,
`generationConfig`) in user or project `settings.json`. dsh, for a model that declares reasoning, sends the system
prompt as `role: "developer"` and the cap as `max_completion_tokens`; ignis accepts both (`template.rs`, `api.rs`
`one_cap`). dsh and pi need a placeholder key against a keyless server. dsh's *native DeepSeek* adapter adds
`dsh_session_log` (on by default; prompts, content and tool I/O) to the body of whatever server `$DEEPSEEK_BASE_URL`
points at; the Custom model API path does not.

Inferences, labelled as such:

- *Inference:* `~/.agents/skills` is read by all four from any working directory, so one global skill folder reaches
  every harness with no per-harness code, if its frontmatter keeps to what all four read: `name` and `description`
  (`disable-model-invocation` is read by pi, dsh and qwen-code, not opencode). A project `.agents/skills` at the repo
  root reaches qwen-code only when it is launched from the repo root.
- *Inference:* one MCP server over Streamable HTTP with a static `Authorization` header is the widest common MCP
  configuration: all four take it. In pi its tools are declared to the model only with the server's
  `exposure: direct`; the default `codemode`, like `deferred` (reached through `tool_search`), keeps them out of the
  model's tool list. qwen-code defers them behind `tool_search` too unless the server is configured with
  `alwaysLoadTools: true`.
- *Inference:* three of the four never gate a `curl` to `127.0.0.1` by default, so a skill that tells the model to
  call `PATCH /v1/config` runs unprompted there, and the only boundary is ignis's own API key. In qwen-code the same
  call is judged by the auto-mode LLM classifier; with ignis as the only model configured, ignis judges its own
  configuration change. That verdict was not run.
- *Inference:* only a native code extension can do what is harness-specific: register a System One classifier in pi's
  own model registry, rewrite opencode's system prompt (the date line), or draw a generated image in dsh's Web UI.
  qwen-code has no such extension; its "native" connector is the same MCP server and skill, packaged as a
  `qwen-extension.json` bundle (which can also carry hooks and the settings the MCP server reads as env vars).

## Implications

- For the connector design: the shared mechanisms (Agent Skills in `~/.agents/skills`, an MCP server) cover all four
  harnesses at once; native code plugins exist for three of them, as three TypeScript codebases against three
  extension APIs that all shipped breaking changes in the last months.
- A tool that returns a generated image helps only a model entry declared image-capable. For a text-only model entry
  a tool should return the file's path, not the image; in opencode an image must be a base64 `data:` URL.
- MCP tools that run as long as an image generation need a timeout set (opencode: `timeout`, otherwise the MCP SDK's
  default; dsh: `toolCallTimeoutMs`, 60 s; pi: `timeout`, 60 s; qwen-code: `timeout`, 600 s by default) or progress
  notifications.
- A pi connector for decisions can reuse pi's System One client unchanged for text; image decisions need checking
  first (Limits).
- Prompt reuse: opencode's system prompt changes once a day (the date line) and nothing but a plugin removes it;
  qwen-code's startup message carries the date, a folder tree and a git snapshot, and its system prompt changes
  whenever auto-memory writes.
- qwen-code side jobs: with `fastModel` unset, side queries and the background memory agents land on the main ignis
  model as extra requests; set to another id, `fastModel` is a second model id, which makes ignis switch models.
- The agent path and [Connect a client](../user/start/connect-a-client.md) already document the provider pitfalls
  (one model id, a stable system prompt); a connector page would extend them, not replace them.

## Limits and unknowns

- **Image decisions from pi to ignis are unverified.** For a non-OpenRouter base URL pi sends images as an `assets`
  map with `state: ["<image-1>", state]` and a `TypeSafe-Danger-Zone-Images` header. ignis's `DecideRequest` takes
  images as OpenAI content parts in `state` and has no `assets` field. *Inference:* with pi's default the image does
  not reach ignis's model; `compat.imageInput: "state-image-urls"` puts `image_url` parts before the state, and
  whether ignis parses that mixed array as content parts was not checked.
- Nothing was run: no harness was exercised against ignis (opencode and qwen-code are installed here but were not
  run). Permission behaviour is read from docs and source.
- opencode: whether a v2 plugin takes effect in the shipped binary; whether a v1 plugin's `config` hook mutation
  persists; how `@ai-sdk/openai-compatible` serializes the synthetic image message; the MCP SDK's default tool-call
  timeout; whether opencode has any transcription client.
- pi: whether an extension can keep its own keys in `settings.json`. Its author's 2025 "no MCP" post and Earendil's
  2026-09-29 reversal were read through a summarising fetch; re-check the wording before quoting.
- dsh: whether the Electron app matches the npm build; how an MCP tool's input schema with unsupported keywords is
  treated; tool-call behaviour over `openai-completions` is pi-ai's and was not traced.
- qwen-code: the auto-mode classifier's real verdict on a `curl` to ignis, and whether ignis can serve as that
  classifier (its structured Stage 1 output and latency budget); whether a `discoveryCommand` tool can return images;
  whether a Docker or Podman sandbox reaches the host's loopback; how much of each surface is inherited from Gemini
  CLI (no diff was made). Facts were re-checked on 0.23.0 only for the files the research names.
- All four ship often; facts hold for the commits named in Evidence.

## Follow-ups

- The connector design (one shared mechanism or a plugin per harness, where it lives, how a config-changing tool is
  authorised) is to be settled by a grilling of this finding
  ([#383](https://github.com/gpillon/ignis/issues/383)); the decision goes to an ADR.
