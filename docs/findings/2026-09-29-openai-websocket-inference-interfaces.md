# OpenAI's WebSocket inference standard is Responses WebSocket mode, and it presupposes streaming Responses

- Kind: research
- Status: current
- Observed: 2026-09-29
- Last verified: 2026-09-29
- Scope: serving / the `/v1` surface, `POST /v1/responses`, a WebSocket transport; clients Codex CLI, openai-python, openai-node, openai-agents
- Related: [ADR 0029](../adr/0029-cross-request-state-reuse.md) (content-matched reuse, no session id),
  [ADR 0036](../adr/0036-the-v1-surface-documents-itself.md) (the `/v1` path set),
  [spec server/01](../specs/server/01-openai-http.md) (`/v1/responses` non-streaming in v1),
  raw source copies in `.scratch/openai-websocket-research-2026-09-29/raw/` (untracked)
- Superseded by: none

## Question

Which WebSocket interface does OpenAI define for text LLM inference, what exactly
does a server have to speak to be compatible with it, who uses it, and what does
ignis lack today to serve it?

## Evidence

All facts below come from primary sources: OpenAI doc pages downloaded as their
`.md` twins, the `openai-openapi` spec, and source files from openai-python,
openai-node, openai-agents-python, codex, vLLM, SGLang and Ollama. The local
copies are in `.scratch/openai-websocket-research-2026-09-29/raw/`. Every claim
carries its source tag.

| Tag | URL |
|---|---|
| [WSG] | https://developers.openai.com/api/docs/guides/websocket-mode |
| [WSE] | https://developers.openai.com/api/reference/resources/responses/websocket-events |
| [WSE-beta] | https://developers.openai.com/api/reference/resources/beta/subresources/responses/websocket-events |
| [STEER] | https://developers.openai.com/api/docs/guides/steering |
| [CL] | https://developers.openai.com/api/docs/changelog |
| [OAS] | https://github.com/openai/openai-openapi/blob/main/openapi.yaml (schemas `ResponsesClientEvent`, `ResponsesClientEventResponseCreate`, `ResponsesServerEvent`, `ResponseWsError`) |
| [PY] | https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py |
| [PY-client] | https://github.com/openai/openai-python/blob/main/src/openai/_client.py |
| [PY-reconn] | https://github.com/openai/openai-python/blob/main/src/openai/types/websocket_reconnection.py |
| [NODE] | https://github.com/openai/openai-node/blob/master/src/resources/responses/ws.ts and `ws-base.ts` |
| [AGENTS] | https://github.com/openai/openai-agents-python/blob/main/docs/models/index.md (rendered: https://openai.github.io/openai-agents-python/models/) |
| [CODEX-client] | https://github.com/openai/codex/blob/main/codex-rs/core/src/client.rs |
| [CODEX-api] | https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/common.rs |
| [CODEX-mpi] | https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs |
| [CODEX-cfg] | https://developers.openai.com/codex/config-reference |
| [RT] | https://developers.openai.com/api/docs/guides/realtime |
| [RT-WS] | https://developers.openai.com/api/docs/guides/realtime-websocket (also served at `/api/docs/guides/voice-websockets`) |
| [RT-CONV] | https://developers.openai.com/api/docs/guides/realtime-conversations |
| [RT-CE] | https://developers.openai.com/api/reference/resources/realtime/client-events |
| [RT-SE] | https://developers.openai.com/api/reference/resources/realtime/server-events |
| [RT-CS] | https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create |
| [RT-WEBRTC] | https://developers.openai.com/api/docs/guides/realtime-webrtc |
| [RT-SIP] | https://developers.openai.com/api/docs/guides/realtime-sip |
| [RT-SC] | https://developers.openai.com/api/docs/guides/realtime-server-controls |

### A. Responses API WebSocket mode

#### A.1 Launch, purpose, stated latency rationale

- Launched **2026-02-23**: "Launched WebSocket mode for the Responses API." [CL]
- Purpose: "The Responses API supports a WebSocket mode for long-running, tool-call-heavy workflows. Beyond lowering latency, `stream_id` enables WebSocket multiplexing: one persistent connection to `/v1/responses` can run parallel conversations and fork an existing conversation onto a new stream. Continue each turn by sending only new input items plus `previous_response_id`." [WSG]
- Latency rationale: "Because the connection stays open and each turn sends only incremental input, WebSocket mode reduces per-turn continuation overhead and improves end-to-end latency across long chains. For rollouts with 20+ tool calls, we have seen up to roughly 40% faster end-to-end execution." [WSG]
- Its main win is a server-side cache: "On an active WebSocket connection, the service keeps recent previous-response state in a connection-local in-memory cache. When you use `stream_id`, each lane keeps its latest cached response, so continuing from the latest response in that lane is fast because the service can reuse connection-local state." [WSG]
- It is "the Responses API over websocket transport, not the Realtime API". [AGENTS]

#### A.2 URL and auth

- URL: `wss://api.openai.com/v1/responses`. The guide prose names only the path ("one persistent connection to `/v1/responses`" [WSG]). Both SDKs build the URL from the HTTP base URL: Python `_prepare_url` swaps the scheme to `wss` (or `ws` for `http`) and appends `/responses` to `base_url` (default `https://api.openai.com/v1`), and uses `websocket_base_url` instead when one is set. [PY] [PY-client]
- Auth: an `Authorization: Bearer <key>` header on the upgrade request. Python sends `self.__client.auth_headers` plus default and extra headers as `additional_headers` on the handshake, with `security={"bearer_auth": True}` [PY]. `_bearer_auth` returns `{"Authorization": f"Bearer {api_key}"}` [PY-client]. Node passes `_buildWebSocketHeaders(authHeaders)` as `ws` headers with `followRedirects: false`. [NODE]
- The OpenAI SDKs add no `OpenAI-Beta` header for GA WebSocket mode (none appears in [PY] `_connect_ws`). Codex does send one, `OpenAI-Beta: responses_websockets=2026-02-06` (`RESPONSES_WEBSOCKETS_V2_BETA_HEADER_VALUE`). [CODEX-client]
- Nothing documents subprotocol-based or query-param auth for Responses WS (see Unverified).

#### A.3 Client → server events

Two GA client events: `response.create` and `response.steer`. The beta surface adds `response.inject`. [WSE] [WSE-beta] [OAS `ResponsesClientEvent` = `anyOf` `ResponsesClientEventResponseCreate` | `ResponseSteerEvent`]

#### `response.create`

> "Client event for creating a response over a persistent WebSocket connection. This payload uses the same top-level fields as `POST /v1/responses`, plus WebSocket-only envelope metadata.
> Notes:
> - `stream` is implicit over WebSocket and should not be sent.
> - `background` is not supported over WebSocket.
> - `stream_id` is WebSocket-only and is not part of `POST /v1/responses`." [WSE] (same text in [OAS])

- Shape: the fields sit flat at the top level next to `type`. There is **no** nested `response` object as in Realtime. The schema is `allOf` of `{type, stream_id}` and `CreateResponse`, the same schema as the HTTP body. [OAS]
- Fields listed in [WSE]: `type`, `access_programs`, `background`, `context_management`, `conversation`, `include`, `input`, `instructions`, `max_output_tokens`, `max_tool_calls`, `metadata`, `model`, `moderation`, `parallel_tool_calls`, `previous_response_id`, `prompt`, `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, `reasoning`, `safety_identifier`, `service_tier`, `store`, `stream`, `stream_id`, `stream_options`, `temperature`, `text`, `tool_choice`, `tools`, `top_logprobs`, `top_p`, `truncation`, `user`. `stream` and `background` stay in the schema even though the notes say not to use them.
- `stream_id` constraints: `minLength: 1`, `maxLength: 256`, `pattern: ^[A-Za-z0-9_.-]+$` [OAS]. "An empty string is not a valid `stream_id`; omit the field to select the default lane." [WSG]
- Reference example [WSE]:

```json
{
  "type": "response.create",
  "stream_id": "agent_1",
  "model": "gpt-6-astra",
  "input": "Say hello."
}
```

- Guide example with message input [WSG] (Node payload):

```javascript
ws.send({
  type: "response.create",
  stream_id: "main",
  model: "gpt-6-astra",
  store: false,
  input: [
    {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "Find fizz_buzz()" }],
    },
  ],
  tools: [],
});
```

- **Warmup, `generate: false`**: "Clients can optionally warm up request state by sending `response.create` with `generate: false`. [...] `generate: false` does not return a model output, but prepares request state so the next generated turn can start faster. The warmup request returns a response ID that you can chain from with `previous_response_id`, including on later turns in a response chain." [WSG]. Codex uses it: "WebSocket prewarm is a v2-only `response.create` with `generate=false`; it waits for completion so the next request can reuse the same connection and `previous_response_id`." [CODEX-client]. **Discrepancy:** `generate` is missing from the [WSE] field list and from [OAS] `ResponsesClientEventResponseCreate`. Only the guide prose and Codex's `ResponseCreateWsRequest.generate: Option<bool>` [CODEX-api] mention it.

#### `response.steer` (mid-turn steering, GPT-6 family)

- "Queues user input to steer a response on this WebSocket connection. [...] This event accepts only `type`, `previous_response_id`, and `input`. Do not send `stream_id`; the target response determines the WebSocket lane." [WSE]
- "Mid-turn steering is available with the GPT-6 model family over a WebSocket". Steering "does not rewrite output already sent [...] undo earlier actions, or cancel tools that have already started." [STEER]
- Example [WSE]:

```json
{
  "type": "response.steer",
  "previous_response_id": "resp_123",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [{ "type": "input_text", "text": "Prioritize the database rollout." }]
    }
  ]
}
```

- If steering interrupts the running response, it "ends with `response.incomplete` and `incomplete_details.reason: "steered"`", followed by an automatic successor `response.created`. [STEER] [WSE]
- Steer error codes: `invalid_input`, `steering_not_supported`, `response_not_found`, `too_many_pending_steers`. [STEER]

#### `response.inject` (beta only)

- "Injects input items into an active response over a WebSocket connection. The items are validated and committed atomically. Currently, the server accepts client-owned tool outputs that resume a waiting agent." [WSE-beta]
- Server replies: `response.inject.created` or `response.inject.failed` (codes `response_already_completed`, `response_not_found`). [WSE-beta]
- Used by the Agents SDK's experimental hosted multi-agent model, which "sends the `OpenAI-Beta: responses_multi_agent=v1` WebSocket header" and requires `client.beta.responses.connect` (`openai[realtime]` >= 2.45.0). [AGENTS]

#### A.4 Server → client events

- **Shared events are the SSE events.** "These events use the same payloads over WebSocket and [HTTP streaming](https://developers.openai.com/api/reference/resources/responses/streaming-events)." [WSE] "Events within each response follow the existing Responses streaming event model. Events from different lanes can interleave." [WSG]
- The one addition on shared events is `stream_id`: "For named streams, server events include the matching `stream_id`, including terminal events and request-scoped errors. If you omit `stream_id`, the request uses an implicit default lane, and its events do not include `stream_id`." [WSG]. [OAS] `ResponsesServerEvent` wraps each SSE event schema `allOf` with an optional `stream_id`.
- Shared server events listed in [WSE], in order: `response.created`, `response.in_progress`, `response.completed`, `response.failed`, `response.incomplete`, `response.output_item.added`, `response.output_item.done`, `response.content_part.added`, `response.content_part.done`, `response.output_text.delta`, `response.output_text.done`, `response.refusal.delta`, `response.refusal.done`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.file_search_call.{in_progress,searching,completed}`, `response.web_search_call.{in_progress,searching,completed}`, `response.reasoning_summary_part.{added,done}`, `response.reasoning_summary_text.{delta,done}`, `response.reasoning_text.{delta,done}`, `response.image_generation_call.{completed,generating,in_progress,partial_image}`, `response.mcp_call_arguments.{delta,done}`, `response.mcp_call.{completed,failed,in_progress}`, `response.mcp_list_tools.{completed,failed,in_progress}`, `response.code_interpreter_call.{in_progress,interpreting,completed}`, `response.code_interpreter_call_code.{delta,done}`, `response.output_text.annotation.added`, `response.queued`, `response.custom_tool_call_input.{delta,done}`, `response.audio.{delta,done}`, `response.audio.transcript.{delta,done}`, `response.compaction.compacting`, `response.shell_call_command.{added,delta,done}`, `response.shell_call_output_content.{delta,done}`.
- **WebSocket-only server events** ("Server events (WebSocket only)") [WSE]: `response.steer.accepted`, `response.steer.pending`, `response.steer.failed`, `error`. Beta adds `response.inject.created` and `response.inject.failed`. [WSE-beta]
- `response.output_text.delta` example [WSE] (identical to SSE):

```json
{
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "delta": "In",
  "sequence_number": 1,
  "logprobs": []
}
```

- `response.completed` carries the full `response` object (`id`, `object: "response"`, `status: "completed"`, `output[]`, `previous_response_id`, `store`, `usage` with `input_tokens_details.cached_tokens`, and so on) plus `sequence_number`. [WSE]
- Terminal events that clients check for in the official examples: `response.completed`, `response.failed`, `response.incomplete`, `error`. [WSG]

#### A.5 Error events

- Schema `ResponseWsError`: `type: "error"`, `error: {code, message, param, type, headers?, misalignment?}`, optional `sequence_number`, optional `status` ("The HTTP status code associated with a WebSocket protocol error."), optional `stream_id`. [WSE] [OAS]
- The WS error shape differs from the HTTP error body: it is an event envelope with `status` at the top level.
- Documented codes, with examples quoted from [WSG]:

```json
{
  "type": "error",
  "status": 400,
  "stream_id": "main",
  "error": {
    "type": "invalid_request_error",
    "code": "previous_response_not_found",
    "message": "Previous response with id 'resp_abc' not found.",
    "param": "previous_response_id"
  }
}
```

```json
{
  "type": "error",
  "status": 400,
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_stream_id",
    "message": "The 'stream_id' field must be a non-empty string with at most 256 characters and may only contain letters, numbers, underscores, hyphens, and periods.",
    "param": "stream_id"
  }
}
```

```json
{
  "type": "error",
  "status": 400,
  "stream_id": "agent_33",
  "error": {
    "type": "invalid_request_error",
    "code": "websocket_stream_limit_reached",
    "message": "This WebSocket connection has reached its maximum number of distinct stream IDs (32). Reuse an existing stream_id or open a new WebSocket connection.",
    "param": "stream_id"
  }
}
```

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "code": "websocket_connection_limit_reached",
    "message": "Responses websocket connection limit reached (60 minutes). Create a new websocket connection to continue."
  },
  "status": 400
}
```

- Scope: "When the server can associate an error with a named lane, the error event includes `stream_id`. Other lanes can continue after a request-scoped error." [WSG]. In the SDK examples, an `error` with no `stream_id` is treated as a connection-level error. [WSG]

#### A.6 `previous_response_id`, incremental input, `store`

- Continuation: "send another `response.create` with: `previous_response_id` set to the prior response ID. `input` containing only new items (for example, tool outputs and the next user message)." [WSG]
- "WebSocket mode uses the same `previous_response_id` chaining semantics as HTTP mode, but it adds a lower-latency continuation path on the active socket." [WSG]
- Cache miss [WSG]:
  - "With `store=true`, the service may hydrate older response IDs from persisted state when available. Continuation can still work, but it loses the in-memory latency benefit."
  - "With `store=false` (including ZDR), there is no persisted fallback. If the ID is uncached, the request returns `previous_response_not_found`."
- ZDR: "Because the service retains previous-response state only in memory and does not write it to disk, you can use WebSocket mode in a way that is compatible with `store=false` and Zero Data Retention (ZDR)." [WSG]
- `store` defaults to true when omitted ("Defaults to true when omitted."). [WSE]
- Eviction on error: "If a same-lane continuation returns a `4xx` or `5xx`, the service evicts the referenced `previous_response_id` from the connection-local cache. A cross-lane fork that returns an error preserves the shared parent so the source lane can continue." [WSG]
- "Reusing a `stream_id` without `previous_response_id` starts a new response; it does not continue the conversation." [WSG]
- Compaction: with server-side compaction (`context_management` with `compact_threshold`), continue normally. With standalone `/responses/compact`, "Start a new chain by omitting `previous_response_id` or setting it to `null`. Pass the compacted output as-is; do not prune the returned window." [WSG]
- Codex (client behaviour, not an API rule): it sends an incremental continuation only when the new request's non-input settings (`model`, `instructions`, `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning`, `store`, `stream`, `include`, `service_tier`, `prompt_cache_key`, `text`) equal the previous request's. Otherwise it sends the full input. [CODEX-client ~l.370-390]

#### A.7 Returning function-call outputs

Tool results go in the **next `response.create`** on the same socket (and lane), with `previous_response_id` set to the response that emitted the `function_call`. There is no separate tool-output event in GA; `response.inject` is beta and only for hosted agents. [WSG]

```javascript
ws.send({
  type: "response.create",
  stream_id: "main",
  model,
  store: false,
  previous_response_id: first.id,
  input: [
    {
      type: "function_call_output",
      call_id: call.call_id,
      output: JSON.stringify(result),
    },
    { role: "user", content: "Now optimize it." },
  ],
  tools,
  tool_choice: "none",
});
```

With steering queued and the response waiting on a client tool, the server emits `response.steer.pending` with `required_input` stubs, e.g. `{"type": "function_call_output", "call_id": "call_789", "name": "lookup"}`. "Return the required input with `response.create` on the same connection, setting `previous_response_id` [...] Do not repeat the accepted steering." [STEER] [WSE]

#### A.8 Concurrency, queueing, lanes

- "Requests with the same `stream_id` stay first-in, first-out and do not overlap. Requests with different `stream_id` values can run concurrently." [WSG]
- Per-connection limits [WSG]:
  - "A connection can have up to 16 active, in-flight responses across named and default lanes. The connection accepts more `response.create` events and queues them until an active response finishes."
  - "A connection accepts up to 32 distinct named `stream_id` values. The implicit default lane does not count toward this named-stream limit."
- Fork: "To branch from a completed response, send its ID as `previous_response_id` with a new `stream_id`." With `store=false`, "Wait for the fork lane to emit `response.in_progress` before advancing the source lane, or retry with `previous_response_id` set to `null` and replay full input context." [WSG]
- **Discrepancy:** the Agents SDK doc says "The [Responses API WebSocket service](https://developers.openai.com/api/docs/guides/websocket-mode) processes one response at a time on each connection and limits each connection to 60 minutes. Open a new connection after that limit; use multiple connections when you need parallel runs." [AGENTS l.235]. The rendered page also says "The service keeps only the most recent response in connection-local memory." This matches the pre-`stream_id` behaviour. The current API guide and reference ([WSG], [WSE], [OAS]) document multiplexing, so the Agents doc looks stale. Default-lane behaviour is still one-at-a-time FIFO in both descriptions.
- Codex never sets `stream_id` (its `ResponseCreateWsRequest` has no such field), so it uses the default lane. [CODEX-api]

#### A.9 Cancellation

- No cancel client event exists for the Responses WebSocket. The only client events are `response.create` and `response.steer` (plus beta `response.inject`). [WSE] [OAS]
- `background` "is not supported over WebSocket" [WSE], and the HTTP cancel endpoint (`POST /v1/responses/{id}/cancel`) is documented for background responses, so it does not apply here. See Unverified.

#### A.10 Connection lifetime and reconnection

- "Connections last up to 60 minutes. Reconnect at the limit." [WSG]. Hitting the limit produces `websocket_connection_limit_reached`. [WSG]
- "When a connection closes (or hits the 60-minute limit), its connection-local cache disappears for every lane." [WSG]. Recovery options: continue with `previous_response_id` if `store=true`; otherwise start fresh with `previous_response_id: null` and full context; or seed from `/responses/compact` output. [WSG]
- "Queued steering input exists only on the current connection; it isn't stored with the original response." [STEER]
- SDK auto-reconnect retries only these close codes: 1001, 1005, 1006, 1011, 1012, 1013, 1015 [PY-reconn]. Python `connect(... max_retries=5, initial_delay=0.5, max_delay=8.0, max_queue_size=1_048_576)` [PY]. Node: "Maximum number of reconnection attempts. Default: 5.", and reconnect is on only when `reconnect` is non-null. [NODE ws-base.ts]

#### A.11 SDK exposure

- Install: `pip install "openai[realtime]>=3.8.0"`, `npm install openai@^7.10.0 ws`, `gem install openai async-websocket`. [WSG] "The .NET SDK does not provide a Responses WebSocket client". [STEER]
- **openai-python**: `client.responses.connect(extra_query=..., extra_headers=..., websocket_connection_options=..., on_reconnecting=..., max_retries=5, ...)` returns `ResponsesConnectionManager`. As a context manager it yields `ResponsesConnection`, with methods `send`, `send_raw`, `recv`, `recv_bytes`, `close`, `parse_event`, `on/off/once`, `dispatch_events`, `__iter__`. The typed helpers are `connection.response.create(...)` and `connection.response.steer(input=..., previous_response_id=...)`. The async twins are `AsyncResponsesConnection` / `AsyncResponsesConnectionManager`. It uses the `websockets` package (`websockets.sync.client.connect`). [PY]
- **openai-node**: `import { ResponsesWS } from "openai/resources/responses/ws"; const ws = new ResponsesWS(client); ws.send({...}); for await (const event of ws) {...}` or `ws.stream()`. Events arrive wrapped as `{type: "message", message}` / `{type: "error", error}` / `reconnecting` / `reconnected`. `ws.close()`. It needs the `ws` package. [NODE] [WSG]
- **openai-agents-python**: "By default, OpenAI Responses API requests use HTTP transport." Opt in with `set_default_openai_responses_transport("websocket")`, `OpenAIProvider(use_responses_websocket=True, websocket_base_url=..., responses_websocket_options={...})`, `MultiProvider(openai_use_responses_websocket=True)`, or the `responses_websocket_session()` helper. Model classes: `OpenAIResponsesWSModel` (websocket) and `OpenAIResponsesModel` (HTTP). [AGENTS]
- **Codex CLI**: WebSocket is **on by default** for the built-in `openai` provider (`create_openai_provider` sets `supports_websockets: true`). The OSS providers (`ollama`, `lmstudio`) and generic ones default to `false`. [CODEX-mpi]. It is configurable per provider: "`model_providers.<id>.supports_websockets` — Whether that provider supports the Responses API WebSocket transport." [CODEX-cfg]
- Codex falls back to HTTP for the session on WS failure. It counts `codex.transport.fallback_to_http`, and an HTTP **426 Upgrade Required** on the upgrade triggers `try_switch_fallback_transport` (`client.rs` ~l.1507 and ~l.1922). [CODEX-client]
- Codex payload shape (`ResponseCreateWsRequest`, serialized as `{"type":"response.create", ...}`): `model`, `instructions`, `previous_response_id`, `input`, `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning`, `store`, **`stream`** (always serialized), `stream_options`, `include`, `service_tier`, `prompt_cache_key`, `text`, `generate`, `client_metadata`, `access_programs`. [CODEX-api]. So a compatible server must accept and ignore `stream`, even though [WSE] says it "should not be sent".
- Codex also sends extra handshake headers: `OpenAI-Beta: responses_websockets=2026-02-06`, `x-client-request-id`, `x-codex-turn-state`, `x-codex-routing-hint`, and so on. [CODEX-client]

---

### B. Realtime API (GA) — the text-only path

#### B.1 Status, URL, auth

- GA announced 2025-08-28: "The OpenAI Realtime API is now generally available." [CL]
- Beta removed 2026-05-12: "The Realtime API Beta was deprecated and removed from the API on May 12, 2026." [CL]
- URL: `wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1` [RT-WS]. Sideband control of an existing WebRTC or SIP call uses `wss://api.openai.com/v1/realtime?call_id={call_id}` [RT-SC] [RT-SIP]. Python `client.realtime.connect(model=..., call_id=...)` adds `call_id` as a query parameter. (openai-python `src/openai/resources/realtime/realtime.py`)
- Server auth: `Authorization: Bearer $OPENAI_API_KEY` header, plus an optional `OpenAI-Safety-Identifier` header. [RT-WS] [RT]
- Browser auth over WebSocket subprotocols [RT-WS]:

```javascript
const ws = new WebSocket(
  "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1",
  [
    "realtime",
    // Use a short-lived token fetched from your application server.
    "openai-insecure-api-key." + OPENAI_REALTIME_EPHEMERAL_KEY,
    // Optional
    "openai-organization." + OPENAI_ORG_ID,
    "openai-project." + OPENAI_PROJECT_ID,
  ]
);
```

- `OpenAI-Beta: realtime=v1`: "Remove the `OpenAI-Beta: realtime=v1` header when calling the GA interface." [RT]. The legacy beta SDK path still sets it (`"OpenAI-Beta": "realtime=v1"` in openai-python `src/openai/resources/beta/realtime/realtime.py`).
- Ephemeral client secrets: `POST /v1/realtime/client_secrets`. "Client secrets are short-lived tokens that can be passed to a client app [...] The client secret is a string that looks like `ek_1234`." `expires_after.seconds` accepts "a value between `10` and `7200` (2 hours). This default to 600 seconds (10 minutes) if not specified." A secret "can be used to create multiple sessions until it expires." [RT-CS]
- Session duration: "The maximum duration of a Realtime session is **60 minutes**." [RT-CONV]

#### B.2 Client events (GA) [RT-CE]

Full list: `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`, `input_audio_buffer.clear`, `conversation.item.create`, `conversation.item.retrieve`, `conversation.item.truncate`, `conversation.item.delete`, `response.create`, `response.cancel`, `output_audio_buffer.clear`. Every client event takes an optional `event_id`.

Text-only usage:

1. `session.update`. "Only the fields that are present in the `session.update` are updated." `session.type` is required in GA (`"realtime"`). Set `output_modalities: ["text"]` for text without audio ("set to ["text"] if you want text without audio"). [RT-CE] [RT-CONV]

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "instructions": "You are a creative assistant that helps with design tasks.",
    "tools": [ { "type": "function", "name": "display_color_palette", "description": "...", "parameters": { "type": "object", "properties": { } } } ],
    "tool_choice": "auto"
  }
}
```

(The `tools` entry is abbreviated from the [RT-CE] example.)

2. `conversation.item.create` with a user text message [RT-CE]:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "user",
    "content": [{ "type": "input_text", "text": "hi" }]
  }
}
```

3. `response.create`. Its config is nested under `response` (unlike Responses WS), and "If these are set, they will override the Session's configuration for this Response only." [RT-CE]

```json
{"type": "response.create", "response": {"output_modalities": ["text"]}}
```

(text-only form from [RT-CONV]). The out-of-band form uses `"conversation": "none"` plus an explicit `input` with `item_reference`s. [RT-CE]

4. `response.cancel`: "Send this event to cancel an in-progress response. The server will respond with a `response.done` event with a status of `response.status=cancelled`. If there is no response to cancel, the server will respond with an error." Optional `response_id`. [RT-CE]

```json
{ "type": "response.cancel", "response_id": "resp_12345" }
```

- Function results: `conversation.item.create` with `{"type": "function_call_output", "call_id": ..., "output": "<json string>"}`, then another `{"type": "response.create"}`. [RT-CONV]
- Concurrency: "Only one Response can write to the default Conversation at a time, but otherwise multiple Responses can be created in parallel. The `metadata` field is a good way to disambiguate multiple simultaneous Responses." [RT-CE]
- VAD is on by default. For text-driven control, set `turn_detection` to `null`, or set `turn_detection.create_response`/`interrupt_response` to `false`. [RT-CONV]

#### B.3 Server events (GA) [RT-SE]

Full list: `error`, `session.created`, `session.updated`, `conversation.item.added`, `conversation.item.done`, `conversation.item.retrieved`, `conversation.item.input_audio_transcription.{completed,delta,segment,failed}`, `conversation.item.truncated`, `conversation.item.deleted`, `input_audio_buffer.{committed,dtmf_event_received,cleared,speech_started,speech_stopped,timeout_triggered}`, `output_audio_buffer.{started,stopped,cleared}`, `response.created`, `response.done`, `response.output_item.{added,done}`, `response.content_part.{added,done}`, `response.output_text.{delta,done}`, `response.output_audio_transcript.{delta,done}`, `response.output_audio.{delta,done}`, `response.function_call_arguments.{delta,done}`, `response.mcp_call_arguments.{delta,done}`, `response.mcp_call.{in_progress,completed,failed}`, `mcp_list_tools.{in_progress,completed,failed}`, `rate_limits.updated`, `conversation.created`, `conversation.item.created`.

Order of text-generation events [RT-CONV]: `conversation.item.added` → `conversation.item.done` → `response.created` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta` → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → `response.done` → `rate_limits.updated`.

- `session.created`: "Emitted automatically when a new connection is established as the first server event." Example fields: `type`, `event_id`, and `session: {type: "realtime", object: "realtime.session", id: "sess_...", model, output_modalities, instructions, tools, tool_choice, max_output_tokens: "inf", expires_at, audio{...}}`. [RT-SE]
- `response.output_text.delta` [RT-SE]. Unlike the Responses API version it carries `event_id` and `response_id` and has no `sequence_number`:

```json
{
  "event_id": "event_4142",
  "type": "response.output_text.delta",
  "response_id": "resp_001",
  "item_id": "msg_007",
  "output_index": 0,
  "content_index": 0,
  "delta": "Sure, I can h"
}
```

- `response.done`: "Always emitted, no matter the final state. [...] Clients should check the `status` field [...] (`completed`) or [...] `cancelled`, `failed`, or `incomplete`." `response.object` is `"realtime.response"`, and `usage` has `input_token_details.{text_tokens,audio_tokens,image_tokens,cached_tokens}`. [RT-SE]
- `error`: "Most errors are recoverable and the session will stay open". Example [RT-SE]:

```json
{
  "event_id": "event_890",
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_event",
    "message": "The 'type' field is missing.",
    "param": null,
    "event_id": "event_567"
  }
}
```

(`error.event_id` echoes the client event that caused the error. [RT-CONV])

#### B.4 GA vs beta naming

- "use the newer response event names like `response.output_text.delta`, `response.output_audio.delta`, and `response.output_audio_transcript.delta`." [RT]
- Old beta names, confirmed from openai-python's retained beta types: `response.text.delta` (`src/openai/types/beta/realtime/response_text_delta_event.py`, `Literal["response.text.delta"]`) versus GA `response.output_text.delta` (`src/openai/types/realtime/response_text_delta_event.py`). Similarly `response.audio_transcript.delta` in beta (`beta/realtime/response_audio_transcript_delta_event.py`).
- Beta `response.create` used `modalities` (beta `response_create_event.py`: `modalities: Optional[List[Literal["text", "audio"]]]`). GA uses `output_modalities`. [RT-CONV]
- Beta item event `conversation.item.created` (beta `conversation_item_created_event.py`). GA's text flow uses `conversation.item.added`/`conversation.item.done`, although the GA reference still lists `conversation.item.created` and `conversation.created`. [RT-SE]
- Other GA changes: "set `session.type`, move output audio configuration under `session.audio.output`". [RT]

#### B.5 input_audio_buffer events (listed only, audio out of scope)

- Client: `input_audio_buffer.append`, `input_audio_buffer.commit`, `input_audio_buffer.clear`; related `output_audio_buffer.clear`. [RT-CE]
- Server: `input_audio_buffer.committed`, `input_audio_buffer.cleared`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`, `input_audio_buffer.timeout_triggered`, `input_audio_buffer.dtmf_event_received`. [RT-SE]

#### B.6 Alternatives (noted only)

- WebRTC: `POST https://api.openai.com/v1/realtime/calls` (SDP), with ephemeral keys from `/v1/realtime/client_secrets`. [RT-WEBRTC] [RT]
- SIP: `sip:$PROJECT_ID@sip.api.openai.com;transport=tls` (EU: `sip-eu.api.openai.com`), with call control via `https://api.openai.com/v1/realtime/calls/...` and a sideband `wss://api.openai.com/v1/realtime?call_id=...`. [RT-SIP]
- A separate newer API, **GPT-Live**, runs at `wss://api.openai.com/v1/live/sessions`, with `session.start`/`session.started`, `session.input_audio.append`, and a delegated Responses backend. It is distinct from Realtime and not covered here. [RT-WS]

---

### C. Other inference servers

| Server | Responses WS mode | `/v1/realtime` | Evidence |
|---|---|---|---|
| vLLM | **Not merged.** PR #35492 "feat(responses): add WebSocket mode for Responses API", opened 2026-02-27, still OPEN | **Yes, ASR/transcription only.** `@router.websocket("/v1/realtime")` in `vllm/entrypoints/speech_to_text/realtime/api_router.py`. `protocol.py` literals: client `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`; server `session.created`, `transcription.delta`, `transcription.done`, `error`. The `transcription.*` names are vLLM's own, not OpenAI Realtime event names; PCM16 16 kHz | https://github.com/vllm-project/vllm/pull/35492 ; https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/speech_to_text/realtime/api_router.py ; https://docs.vllm.ai/en/latest/examples/speech_to_text/realtime/ |
| SGLang | none found (only HTTP `serving_responses.py`) | **Yes, transcription subset only.** `http_server.py`: `@app.websocket("/v1/realtime")` → `openai_v1_realtime_transcription`, commented "This handler implements the transcription subset only; chat-mode session.update payloads are rejected" | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py ; `python/sglang/srt/entrypoints/openai/realtime/` |
| llama.cpp server | none. README documents HTTP `POST /v1/responses` only | none. Issue #23101 (realtime STT protocol) closed NOT_PLANNED 2026-07-17 | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md ; https://github.com/ggml-org/llama.cpp/issues/23101 |
| Ollama | **Refuses deliberately.** Its Codex proxy answers a WS upgrade on `/api/codex/v1/responses` with **426**, commented "Codex treats 426 as a session-wide fallback to HTTP" | none found | https://github.com/ollama/ollama/blob/main/internal/proxy/codex_desktop.go ; `server/codex_proxy_test.go` (`TestCodexProxyWebSocketUpgradeRequestsHTTPFallback`) |
| LM Studio | nothing found | nothing found | (search only) |
| TGI | nothing found; repo `huggingface/text-generation-inference` is **archived** | nothing found | https://github.com/huggingface/text-generation-inference |
| LocalAI | not confirmed (see Unverified) | **Yes, full pipeline** (VAD+STT+LLM+TTS). Page: "OpenAI Realtime API which enables low-latency, multi-modal conversations (voice and text) over WebSocket", URL `ws://localhost:8080/v1/realtime?model=gpt-realtime`, also `/v1/realtime/calls` | https://localai.io/docs/features/openai-realtime/ |
| LiteLLM (proxy) | Issue #22051 "WebSocket mode support for Responses API" closed NOT_PLANNED | (has realtime passthroughs; not investigated) | https://github.com/BerriAI/litellm/issues/22051 |

**Practical takeaway for a server that wants Codex compatibility without implementing WS:**
answer the `/v1/responses` WebSocket upgrade with HTTP 426. Codex then switches the whole session to HTTP/SSE right away instead of burning its connect retries. The pattern is shown by Ollama's `codex_desktop.go` and implemented in Codex `client.rs` (`StatusCode::UPGRADE_REQUIRED` → `try_switch_fallback_transport`). The client-side alternative is `supports_websockets = false` on a custom `model_providers.<id>`. [CODEX-cfg]

## Finding

- **The standard is the Responses API's WebSocket mode**, launched 2026-02-23
  [CL]: a persistent `/v1/responses` socket carrying `response.create` events
  whose body is the `POST /v1/responses` body, answered by **the same events as
  Responses SSE streaming** [WSE]. The Realtime API is a different protocol:
  audio-first, a session plus a conversation, `response.create` nested under
  `response` [RT-CE]. Its text path exists, but no text client uses it for
  agent loops.
- **Its value is the connection-local cache**: `previous_response_id` plus only
  the new items (tool outputs, the next user message), so a 20+ tool-call run
  is "up to roughly 40% faster" [WSG]. The cache lives only in memory for the
  life of the socket and works with `store=false` [WSG].
- **It presupposes streaming Responses.** Every event a socket carries is a
  Responses SSE event (`response.created`, `response.output_item.added`,
  `response.output_text.delta`, `response.function_call_arguments.delta`,
  `response.completed`, ...). The only socket-only events are `error` (an
  envelope with a top-level `status`) and the `response.steer.*` events [WSE].
- **Codex is the client that uses it**, and Codex has removed
  `wire_api = "chat"` (`codex_mpi.rs`: "`wire_api = \"chat\"` is no longer
  supported"), so Codex against any server means the Responses API. For custom
  providers the WebSocket is opt-in (`supports_websockets` defaults to `false`
  [CODEX-mpi]). What makes Codex able to run on ignis is streaming Responses
  with function tools; the socket is the latency layer on top.
- **Codex's wire habits a server must tolerate:** it always serializes
  `stream`, even though [WSE] says not to send it; it prewarms with
  `generate: false` and waits for completion (undocumented in the schema); it
  never sets `stream_id`; and it treats an HTTP **426** on the upgrade as a
  session-wide switch to HTTP [CODEX-client], which Ollama uses deliberately.
- **Nobody among the open inference servers ships Responses WS mode** as of
  2026-09-29: vLLM's PR #35492 is open, LiteLLM closed its issue as not
  planned, and llama.cpp serves HTTP `/v1/responses` only. vLLM and SGLang serve
  `/v1/realtime` for transcription only.

## Implications

- A WebSocket endpoint in ignis rests on three things `/v1/responses` does not
  have today (spec server/01: non-streaming, `stream: true` is a 400, no
  `tools`, no `previous_response_id`, no `instructions`, no reasoning items):
  the Responses streaming event model, function-call and reasoning items, and
  `previous_response_id` continuation.
- A connection is a session identity. ADR 0029 matches retained state by
  content and deliberately never by session, so a spec has to say how a cached
  `previous_response_id` meets that rule (re-rendering the history and letting
  the content match hit is the rule-preserving reading).
- The upgrade is a `GET` on `/v1/responses`, which is not a `utoipa` operation,
  so ADR 0036's path-set test has to account for it explicitly.
- Dependencies: axum 0.8.9's `ws` feature pulls `tokio-tungstenite` 0.29 /
  `tungstenite` 0.29 (`sha1`, `rand` 0.9). A probe crate on 2026-09-29 showed
  `windows-sys` reached only through `mio` → `tokio`, which is already in the
  graph, so it adds no host chain like #251's `dlltool` trap. A test client
  should pin `tokio-tungstenite` 0.29 to match axum's.

## Limits and unknowns

- **Cancelling a Responses WS response.** No cancel client event is documented ([WSE], [OAS] list only `response.create`, `response.steer`, beta `response.inject`). Whether closing the socket aborts in-flight responses, or whether `POST /v1/responses/{id}/cancel` works on a WS-created non-background response, is **undocumented**.
- **Responses WS auth alternatives.** Only the `Authorization: Bearer` header (via the SDKs) is evidenced. Subprotocol or query-param auth, as Realtime has for browsers, is **not documented** for `/v1/responses`. The exact `wss://api.openai.com/v1/responses` string is derived from SDK URL construction plus the guide's "`/v1/responses`" path. The guide prose never prints the full `wss://` URL.
- **`generate: false`** is documented in guide prose and used by Codex, but **absent** from the [WSE] schema and [OAS]. How a non-generating response looks on the wire (which events, terminal status) is not documented.
- **Is `OpenAI-Beta: responses_websockets=2026-02-06` required?** Codex sends it (it labels the value "V2"). The OpenAI SDKs do not. Whether it changes server behaviour (e.g. enables `generate`) is not documented.
- **Agents SDK vs API guide on concurrency** ("one response at a time" vs 16 in-flight/32 lanes) is reported above. Which reflects production today cannot be settled from docs alone; the API reference is the more authoritative and newer source.
- **Idle timeout / keepalive on the Responses WS server side**: not documented. Only the 60-minute cap is. The Agents SDK defaults `ping_interval`/`ping_timeout` client-side ([AGENTS] example: `{"ping_interval": 20.0, "ping_timeout": 60.0}`).
- **Max message size** for Responses WS frames: not documented server-side. The Agents SDK notes it disables the client-side incoming limit by default (`max_size=None`). [AGENTS]
- **Beta Realtime full event-name mapping**: only the names checked above were confirmed from openai-python beta types (`response.text.delta`, `response.audio_transcript.delta`, `conversation.item.created`, `modalities`). Other renames were not individually verified.
- **LocalAI Responses-WS issue #8644** (seen in search results as "Implement WebSocket Mode support for OpenAI Responses API"): `gh` returns 404 for it, so its status is unverified. The other LocalAI Realtime issues seen (#11103 text-only `modalities` ignored, #11776 `/v1/realtime/calls` GA shape) confirm a Realtime implementation exists.
- **LM Studio**: no primary evidence either way for Responses-WS or `/v1/realtime`.
- **Node Realtime SDK class names** were not checked. Only Python `client.realtime.connect` was verified.

## Follow-ups

- A spec under `docs/specs/` for the Responses WebSocket mode, with streaming
  Responses and function-call items as its prerequisite acceptance criteria.
