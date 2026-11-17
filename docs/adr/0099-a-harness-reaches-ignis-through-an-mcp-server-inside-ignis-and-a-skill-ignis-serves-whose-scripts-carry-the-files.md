---
status: proposed
---

# A harness reaches ignis through an MCP server inside ignis and a skill ignis serves, whose scripts carry the files

On 2026-11-17 (#383) the owner settled how a coding agent's harness (opencode,
qwen-code, pi, DeepSeek Harness) reaches ignis beyond chat; spec #656 builds it.

**The MCP server.** ignis serves the connector's server tools itself, as an
MCP server over Streamable HTTP. It is built on `rmcp`, mounted at `/v1/mcp`
behind the same key as the rest of `/v1`
([ADR 0071](0071-the-server-downloads-and-removes-models-from-the-server-view.md)),
with rmcp's Host check turned off, as no other route has one, and its Origin
list left empty, its default.

- **Stateless.** Every request stands alone: no session, so an ignis restart
  strands no harness. A request's answer is JSON, or an event stream when the
  call sends a notification first, such as the progress a call carrying a
  progress token is sent. There is no server-to-client stream (`GET` is `405`).
- **A tool calls its route through the router, in process,** with the
  caller's `Authorization`. The call meets every gate a client's request
  meets: the key, the serving gate, the count of model use (ADR 0049), the
  body cap. A refusal is the route's own status, error JSON and `Retry-After`.
- **`/v1/mcp` itself is never held by the serving gate.** `initialize` and
  `tools/list` answer during a switch or a warm-up, and the connector can hold
  a call of its own (the image calls, one at a time) instead of meeting a
  `503` at the door; the tool's route is held, or refused, as it is for anyone.

**Where the tools are declared.** A tool is declared where its endpoint is:
an `x-ignis-tool` extension on the endpoint's `#[utoipa::path]` names the
tool, gives its short description for the model and its `read_only` and
`destructive` hints, and narrows its input where the route's own is too wide.
It may fix a query the tool always sends, and mark a tool that changes the
server. The tool list is built once at start from the OpenAPI document,
sorted by name, so it is byte-identical whatever model is loaded, and a
harness's prompt reuse survives it.

**The skill.** The tools that need a file live in a skill, `ignis`, whose
Node scripts run in the harness's shell, beside the files, and call the
existing endpoints:
- generating an image to a file;
- editing an image;
- transcribing audio.

ignis serves the skill from its own build, as it serves the Playground
([ADR 0026](0026-playground-embedded-when-built.md)), so the skill always
matches the server that gave it.

**Why files go through a skill.** MCP has no channel for a client to upload a
file: the model writes a tool's arguments, so a file's bytes would pass
through the model, and a server reading the harness's paths works only on
the same machine. With the scripts, the server touches no file and
[ADR 0073](0073-a-workspace-is-kept-by-the-page-and-ignis-stores-no-files.md)'s
"ignis stores no files" holds.

## Considered Options

- **MCP roots, the server reading the harness's paths.** Rejected: it works
  only when server and harness share a disk, and DeepSeek Harness sends no
  roots.
- **The tool answers with a one-time upload URL and a curl recipe.** Rejected
  by the owner.
- **An upload route, or a file store.** Rejected: ADR 0073.
- **A native plugin per harness.** Rejected: four extension APIs to keep for
  what one MCP server and one skill, which all four read, already give.
- **A separate local connector process.** Rejected: one more binary to
  install on every client machine.
- **The skill as an npm package.** Rejected: an account and a supply chain to
  keep, and a version that drifts from the server's.
- **No skill.** Chosen first, in v1. Reversed once the remote case showed that
  the files can only be handled on the harness's side.
- **Sessions (rmcp's stateful mode).** Not taken: a session id dies with the
  process, and the harnesses recover from that differently or not at all.
  Stateless mode still carries a call's progress.
- **A tool calling its handler directly.** Not taken: it would pass the key,
  the serving gate and the count of model use by, and rebuild each route's
  refusals by hand.

## Consequences

**The tool list:**
- `decide`, `generate_image`, `read_config`, `describe_config` and
  `read_catalog` are always listed.
- `change_config`, `download_model` and `cancel_download` are listed only
  when `connector.allow_changes` is on. It is off by default.
- `connector.enabled` is on by default; off, `/v1/mcp` and the skill's route
  answer 404.
- Both are config fields declared once like any other
  ([ADR 0046](0046-one-declaration-per-config-field.md)), but they take effect
  at start only and cannot be patched, so the list depends on the start
  configuration alone.
- `allow_changes` is not a security boundary: an agent with a shell and the
  one key (ADR 0071) can patch the configuration itself.

**Images:**
- `generate_image` returns a preview, longest edge at most 512 px, and only
  when the text model it returns to has vision on. An image left in the
  harness's history is refused `vision_disabled` without vision, and past
  `vision.max_tokens` with it, on every later turn. With vision off the tool
  answers at once, without switching, and names the skill's script.
- The connector's image calls run one at a time, so several in one turn
  cost one switch out and one back. The vision check reads the text model
  the server switched away from, which it remembers while the image model is
  loaded.
- The image model never runs beside a text model
  ([ADR 0050](0050-ignis-serves-an-image-model.md)). The next chat turn's
  implicit switch brings the text model back: no tool switches back itself.
- The prompt state is lost on the way, until #655 keeps it across a switch.

**Other tools:**
- `decide` answers on the 27B only.
- Transcription is the skill's alone: the speech model runs beside the text
  model ([ADR 0067](0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)),
  but its input is a file.

**Results:**
- Results go as text in `content`. An error is the server's error JSON, with
  `isError` set, then `Retry-After: <seconds>` as a second text when the route
  sent the header. DeepSeek Harness never shows `structuredContent` and drops
  embedded resources.
- The connector never retries.

**Left open:**
- A keyless server on localhost stays open to DNS rebinding, on every route,
  not just this one. Closing it is a server-wide decision, not the
  connector's.
