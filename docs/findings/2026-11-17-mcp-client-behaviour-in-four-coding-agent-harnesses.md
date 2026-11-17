# How opencode, pi, DeepSeek Harness and qwen-code behave as MCP clients: roots, what reaches the model, timeouts, hints, reconnects and server instructions

- Kind: research
- Status: current
- Observed: 2026-11-17
- Last verified: 2026-11-17
- Scope: connectors / the MCP client side of opencode v2.3.0 (and v1.19.4), pi v1.5.2, DeepSeek Harness
  v0.2.2-rc.3 and qwen-code v0.27.4, and the `rmcp` 3.10.0 server transport, for ignis's connector
- Topic: harness-and-build
- Related: [#383](https://github.com/gpillon/ignis/issues/383), [#656](https://github.com/gpillon/ignis/issues/656),
  [Coding-agent harness extension surfaces](2026-10-26-coding-agent-harness-extension-surfaces.md),
  [ADR 0073](../adr/0073-a-workspace-is-kept-by-the-page-and-ignis-stores-no-files.md)
- Superseded by: none
- Summary: The four harnesses differ on roots, images, timeouts and hints. Text in `content` is the only result all of them show; DeepSeek Harness sends no roots, 60 s with no progress; opencode asks on every call; qwen cuts a silent call at 300 s.

## Question

The connector's design (#383, spec #656) needed to know what each harness actually does as an MCP client. The
earlier finding established only that all four are MCP clients. The questions:

- Does a harness send MCP roots?
- What of a tool result reaches the model: an image, a resource link, an embedded resource, `structuredContent`?
- How long does a tool call live, and does progress extend it?
- Does a harness read the tool annotations?
- What happens to its session after the server restarts?
- Where do the server's `instructions` go?
- Does it support prompts and resources?

The answers also cover the Rust server library the connector is built on.

## Evidence

Each harness's source was read at a release tag on 2026-11-17, from shallow clones. Nothing was run. The citations
below are relative to these roots:

| Short name | Release | Commit | Root |
|---|---|---|---|
| OC2 | opencode `v2.3.0` | `a6b15d33` | `packages/core/src` (a rewrite of v1) |
| OC1 | opencode `v1.19.4` | | `packages/opencode/src/mcp` |
| PI | pi `v1.5.2` | `b4afdddf` | |
| DSH | deepseek-harness `dsh-v0.2.2-rc.3` | `38a6a5c9` | `packages/mcp/mcp-client/src` |
| QW | qwen-code `v0.27.4` | `f35dde84` | `packages/core/src` |

For qwen-code, the timeout, 404, replay and `Kind.Read` facts were re-checked at v0.26.1 and v0.23.0. The `rmcp`
facts come from its source at 3.10.0 (2026-11-12).

**Protocol revision.**
- OC2 defaults to `legacy` (`schema/src/mcp.ts`), and qwen-code's remote servers always use legacy
  (`mcp-client.ts:448-460`).
- pi uses `2025-11-25` (`mcp/src/protocol/types.ts:4`).
- DSH negotiates `auto` (`connection.ts:285`).

**Roots.**
- OC2 declares `roots: {}` on legacy connections and answers one root, the Location's directory
  (`mcp/client.ts:357-378`). Its pooled 2026-07-28 connections declare none (`connection.ts:188`). OC1 is the same
  (`index.ts:46,78`).
- PI answers one root, the cwd (`extensions/mcp/runtime.ts:366`; `mcp/src/client.ts:176-218`).
- DSH declares `capabilities: {}`, so no roots (`connection.ts:285`).
- QW answers the cwd plus any added directories (`mcp-client.ts:628-643`; `utils/workspaceContext.ts:47-55`).
- None of the four sends `roots/list_changed`. qwen-code has a sender in `connectToMcpServer` (`:2217-2252`), but
  nothing calls it.

**What reaches the model.**

| | image | resource_link | embedded resource | `structuredContent` |
|---|---|---|---|---|
| OC2 | a `data:` file, moved into a synthetic user message on the AI-SDK path (`tool/mcp.ts:158-174`; `aisdk.ts:509-531`) | bare URI text (`mcp/client.ts:822`) | text kept; a blob of a supported media type becomes media, any other blob a saved file's path | model text only when `content` is empty (`tool/mcp.ts:92-100`; `tool/runtime.ts:274-278`) |
| OC1 | | | | JSON text only when `content` is empty (`catalog.ts:75-79`) |
| PI | an image (`mcp/src/protocol/content.ts:79-101`) | text plus a hint to read it (`extensions/mcp/tools.ts:175-186`) | text kept; an image blob becomes an image, another binary a temp file's path | JSON text only when `content` is empty (`content.ts:112-117`) |
| DSH | an image only if the model declares image input, otherwise a diagnostic | `Resource link: name (uri)` | **dropped**, text too, replaced by a placeholder (`tools.ts:526-527`) | never shown to the model (`tools.ts:283-288`) |
| QW | `inlineData` | `Resource Link: title at uri` | text kept; a blob becomes `inlineData` (`tools/mcp-tool.ts:1764-1795`) | always added as JSON text before the content, unless an identical text block is there (`:1811-1847`; at v0.26.1, not at v0.23.0) |

qwen-code's OpenAI-compatible path has three further behaviours (`core/openaiContentGenerator/`):
- **Tool media moves.** It is stripped from the `tool` message and sent in a follow-up `user` message, under
  `splitToolMedia`, on by default (`converter.ts:820-850`).
- **Images need a declaration.** An image goes out as an `image_url` only when the model's configuration declares
  image input; otherwise a text placeholder goes in its place (`converter.ts:1030-1041`).
- **Old images are evicted.** A request keeps at most 250 images (`image-budget.ts`, `DEFAULT_MAX_REQUEST_IMAGES`).

**Timeouts and progress.**

| | Default | Reset by progress |
|---|---|---|
| OC2 | 12 h a call, `timeout.execution` (`mcp/client.ts:43`) | no, per a source comment (`:724-726`) |
| OC1 | 60 s, MCP SDK 1.29.0's default when `timeout` is unset (`catalog.ts:53-66`; SDK `protocol.ts:106,869-871`) | yes, no total cap |
| PI | 60 s, `timeout` (`runtime.ts:50`) | yes (`docs/mcp.md:68`; `client.ts:536-558`) |
| DSH | 60 s, `toolCallTimeoutMs`; no progress token requested (`tools.ts:155-157`) | no |
| QW | a 600 s wall (`mcp-client.ts:97`; `mcp-tool.ts:745`) and a 300 s idle timer, `QWEN_CODE_MCP_TOOL_IDLE_TIMEOUT_MS` (`config/config.ts:3610-3614`; `mcp-tool.ts:673-731`) | the idle timer only |

**Annotations and permissions.**
- OC2 does not read the hints and asks on every MCP call, by the tool's name (`tool/mcp.ts:59-66`).
- PI passes the hints into the tool definition (`tools.ts:237-247`) but has no permission system; only extensions
  can use them (`core/extensions/types.ts:540`).
- DSH does not read them.
- QW:
  - It asks unless the server has `trust: true` in a trusted folder (`mcp-tool.ts:408-414`).
  - `readOnlyHint` lets a call run in parallel with other reads (`:1337`; `tools/tools.ts:1361`). Any other tool
    runs alone: the scheduler's `isConcurrencySafe` (`core/coreToolScheduler.ts:1752`).
  - Its auto-mode classifier sees the hints marked unverified (`permissions/classifier-prompts/system-prompt.ts:108`).

**After a server restart.**
- OC2 keeps the session id on legacy connections. A `404`, or a `400` "Server not initialized", makes it reconnect
  and retry the request once, tool calls included (`mcp/client.ts:442-460`; `connection.ts:133-150`).
- PI retries once on a new session after a `404`, tool calls included (`transports/streamable-http.ts:385-401`;
  `runtime.ts:280-300`).
- DSH's SDK 2.2.0 raises an error on a `404` and fires `onclose` only from `close()` (SDK
  `streamableHttp.ts:929-936,1029-1031,1119`). DSH reconnects only on `onclose` (`connection.ts:47-52,292`).
- QW re-initializes after a `404`. It replays the call only if the server is trusted and the tool is read-only or
  idempotent; otherwise the call fails and later calls work (`mcp-tool.ts:113-121,369,513-616`). The same holds at
  v0.26.1 and v0.23.0.

**Server `instructions`; prompts and resources.**
- OC2 puts `instructions` in the system prompt, in `<mcp_instructions>`, on every request. The text is frozen at
  session start, and later changes are appended as deltas (`mcp/instructions.ts`; `session/model-request.ts:115-125`).
  Prompts become slash commands; resources are reached through two tools.
- PI shows only the first line, and only for codemode or deferred servers. The full text is available only through
  `describeNamespace` (`docs/mcp.md:71,204,236`). No prompts were found; resources come through three tools.
- DSH puts them in a system-prompt section on every request, capped at 32 KiB (README). It has no prompts;
  resources come through three tools.
- QW adds a `<system-reminder>` user message once per server, and again only if the text changes
  (`core/client.ts:2715-2744`). Prompts become slash commands; resources come through `@server:uri` and a tool.

**rmcp 3.10.0.**
- Licence and toolchain: Apache-2.0 per `Cargo.toml`. Its LICENSE notes the move from MIT, so older contributions
  may stay MIT. It needs Rust 1.88 and edition 2024.
- Mounting: the `transport-streamable-http-server` feature gives `StreamableHttpService`, a tower `Service`
  (`transport/streamable_http_server/tower.rs:1473,1495`), mounted in axum with `nest_service`
  (`examples/servers/src/counter_streamhttp.rs:25-31`).
- Host check:
  - `allowed_hosts` defaults to localhost, `127.0.0.1` and `::1` (`tower.rs:344`).
  - `disable_allowed_hosts()` empties the list (`:363`), and an empty list admits any `Host` (`:1100-1101`).
  - `allowed_origins` defaults to empty (`:345`).
- Protocol: it supports the 2026-07-28 revision by default (`model.rs:176`).

## Finding

Observed:
- **Only text in `content` reaches every model.** DSH never shows `structuredContent` and drops embedded resources.
  OC2 and PI show `structuredContent` only when `content` is empty. QW adds it beside the content.
- **Roots exist in three harnesses, not in DSH.** All three that send them answer only their working directories.
- **The four timeouts do not agree.**
  - DSH cuts a call at 60 s, and nothing extends it.
  - PI and OC1 cut at 60 s unless progress keeps the call alive.
  - QW cuts a call that sends no progress at 300 s.
  - OC2 allows 12 h.
- **Hints are advisory.** OC2 asks on every call regardless. QW still asks unless the server is trusted, but uses
  `readOnlyHint` for parallelism and replay. PI and DSH gate nothing.
- **`instructions` reach all four, in different places.**
  - OC2 and DSH: the system prompt, every request.
  - QW: a reminder in the prefix, once.
  - PI: only the first line, and only in two modes.
- **QW sends a tool's image to the model only when the model's configuration declares image input.** It sends it as
  a user message, not inside the tool message.

Inferred, not observed:
- **DSH after a restart.** Its calls fail until the plugin reloads, and a session negotiated at 2026-07-28 has no
  session id.

## Implications

For any MCP server ignis offers, which today means the connector of #656:
- **Results.** Put the result in `content` as text, the error JSON included. Avoid embedded resources.
- **Stable `instructions`.** Keep them static, so that they stay in the prefix for OC2 and QW.
- **The first line of `instructions`** carries the essentials, for PI.
- **A tool's job fits in 60 s**, or sends progress. That covers PI and OC1, and QW's idle timer. DSH has no way to
  extend a call.
- **A tool returning an image breaks a history it cannot sit in.** The image stays in the harness's history and goes
  back to the server every turn: in a user message for QW, and in a synthetic user message on OC2's AI-SDK path.
- **File access.** Roots let a server read the harness's files only on a shared disk, and never for DSH. Files have
  to move on the harness's side (ADR 0073 kept).
- **A tool's hints never gate access.** An operator switch is the only gate on a server tool.
- **`rmcp` needs its Host check turned off** to be reached behind a tunnel or on a LAN.

## Limits and unknowns

- **Nothing was run.** Every claim is a reading of source or docs at the named commits, and a later release can
  change any of them.
- **Not traced:**
  - OC2's behaviour when the server is down;
  - OC1's reconnect;
  - how opencode declares image input for a custom provider's model, which decides whether an image result reaches
    an ignis model at all. #656's gate checks this.
- **Shallow cases.** DSH's image rule ("only if the model declares image input") and OC1's image handling were read
  shallowly.

## Follow-ups

- The connector's spec and tickets: [#656](https://github.com/gpillon/ignis/issues/656) and its sub-issues. Its gate
  checks the opencode image-input question on the RTX 5090.
