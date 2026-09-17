# Multiple / non-first `system` messages vs strict Qwen templates — engine survey

Researched 2026-09-17. Trigger: agent clients send `[system(big agent prompt), system(short hook line), user, ...]`
(or a system line mid-conversation — Claude Code's `mid-conversation-system-2026-04-07` beta does this,
ollama/ollama#17774). Local traces here are `qwen3.8-27b` (`.scratch/vram-analysis/trace-merged-system.jsonl`).
Commits read: vLLM `a615f53`, llama.cpp `4bc272f`, SGLang `408d233`, Ollama `6b8bef2`, transformers `dbea109` (all 2026-09-16/17).

## 1. What the official Qwen templates say

The check came in with Qwen3.5. Qwen3.6 and Qwen3.8 keep it; nothing newer relaxes it.
`Qwen/Qwen3.5-35B-A3B/chat_template.jinja` L81-86 (the same in Qwen3.5-27B and Qwen3.6-35B-A3B; Qwen3.8-27B L103-107):

```jinja
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
```
`messages[0]` is rendered before the loop (L45-67): with tools it goes into the single tools system block after
`# Tools ... </IMPORTANT>` + `'\n\n' + content`, and without tools it gets its own `<|im_start|>system\n...<|im_end|>`.
A system message at index 1 is `not loop.first`, so the template raises. Two leading system messages are rejected too.

The older templates were permissive. `Qwen/Qwen3-8B` and `Qwen/Qwen3-Next-80B-A3B-Instruct` (`tokenizer_config.json`) render it in place:
`{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}` → `<|im_start|>system\n..<|im_end|>\n`.
**That is exactly ninfer's behaviour** (a separate block in place, no merge, no error). Qwen3-Coder-30B-A3B also renders
`system` roles inside its loop. People work around the strict templates by patching the jinja (e.g. gist garg-aayush/c0211a5f,
HF discussion Qwen3.5-35B-A3B#5, open-webui#22505).

## 2. vLLM (OpenAI server): rejects with HTTP 400. No merge.
- `vllm/entrypoints/chat_utils.py` `parse_chat_messages` (L2160-2196) and `_postprocess_messages` (L2083-2157) only
  normalise content parts and tool-call arguments. Nothing merges or reorders system messages.
- `vllm/renderers/hf.py` `safe_apply_chat_template` L862-880: `except Exception as e: ... raise VLLMValidationError(_template_error_reason(e))`
  with the comment "Surface those as a 400 Bad Request carrying the template's own reason". `_template_error_reason` (L739-749) finds
  the `jinja2.TemplateError`. `vllm/entrypoints/serve/exception_handling/error_response.py` L41-44 turns `VLLMValidationError` into
  `BadRequestError`/400, and L77-81 also maps any raw `TemplateError` to 400.
- vLLM also has a Rust chat renderer (`rust/src/chat/src/renderer/hf/{template,error}.rs`). PR #46976's test output shows the same 400 there.
- The user gets `400 {"error":{"type":"BadRequestError","message":"System message must be at the beginning."}}`.
- It is configurable only by replacing the template: `--chat-template <file>`, or a per-request `chat_template` when started with
  `--trust-request-chat-template` (chat_utils.py L1452). `--chat-template-content-format` changes only string vs OpenAI content-part rendering. It has no effect here.
- Issues: vllm-project/vllm#41114 (open, Qwen3.6-27B). Three fix PRs were all **closed unmerged**: #44505 (merged extra system messages into
  the leading block when the template needs it), #44643 (the same idea in `renderers/hf.py`) and #46976 (patched the test template `qwen35.jinja`).

## 3. llama.cpp server (`--jinja`, the default): rejects. The status is 500 on current builds (400 on some older ones). No merge.
- `common/chat.cpp` @4bc272f applies these workarounds before rendering: `map_developer_role_to_system` (L1263-1266), which makes `developer` → `system`
  and so *creates* more non-first system messages for Codex/Responses clients, and `system_message_not_supported` (L1268-1270, body
  L1016-1030). That merge only runs when the template has **no** system role (`!supports_system_role`). It folds `messages[0]` into
  `messages[1]` with a `"\n"` separator. Qwen3.5+ templates do support system, so it never fires for them.
- Qwen3.5/3.6/3.8 templates contain `<function=`, so they go to the specialized path `common_chat_params_init_qwen3_coder` (chat.cpp L1204-1209,
  called at L1321 outside any try block). The jinja `raise_exception` throws `raised_exception : std::exception`
  (`common/jinja/runtime.h` L705, `value.cpp` L356). `tools/server/server.cpp` `ex_wrapper` L58-67 maps only
  `std::invalid_argument` to 400 and everything else to **500** (`ERROR_TYPE_SERVER`).
  This is confirmed in ggml-org/llama.cpp#27367 (open, 2026-08-18, Qwen3.8-27B): `500 ... Jinja Exception: System message must be at the beginning`
  on both `/v1/chat/completions` and `/v1/messages`.
- Older builds sent these templates through the autoparser try block (L1353-1354 now), which rewraps the error as `invalid_argument`. Those
  builds returned **400** "Unable to generate parser for this template ... System message must be at the beginning" (#20733, #23423, Codex).
- Maintainers' position (#27367 comment): it is the model template's rule, not a llama.cpp bug. Fix PR #27131 is open with no body and no activity.
  Workaround: `--chat-template-file` with a patched template (#27107 and #27367 comments).
- Legacy path (no jinja, `llama_chat_apply_template`): `src/llama-chat.cpp` L250-254 renders ChatML as
  `<|im_start|>role\ncontent<|im_end|>\n` for **every** message in place. There is no check, and it gives the same output as ninfer.

## 4. SGLang: rejects with HTTP 400. No merge.
- `python/sglang/srt/entrypoints/openai/serving_chat.py`: `_CHAT_TEMPLATE_CLIENT_ERRORS = (jinja2.TemplateError, TypeError, ...)` (L35-38).
  L1567-1571 catches them and re-raises `ValueError` ("should be treated as client errors (400 BadRequest)"). `serving_base.py`
  L113-116 then returns `create_error_response` with a default status of 400 (L195-201).
- Message pre-processing only maps `developer`→`system` (L506) and, on some encoder paths, *inserts* an empty system at index 0
  when there is none (L1440-1442). It never merges. No SGLang issue or PR mentions the Qwen error.

## 5. Ollama: renders in place for Qwen3.5/3.6 and merges into one leading system for Qwen3.8. It no longer errors.
Ollama now builds Qwen3.5+ prompts with Go "renderers", not Go templates. Everything below is in `model/renderers/qwen35.go` @6b8bef2.
- Before 0.32.14, the Qwen3.8 renderer rejected these requests with `500 system message must be at the beginning` (#17754, #17774).
- #17757 (merged 2026-08-14) made the renderer tolerate them. Loop L292-296: `if message.Role == "user" || (message.Role == "system" && i != 0)`
  writes `<|im_start|>system\n<content><|im_end|>\n` **in place**. For qwen3.5/3.6 variants this is the current behaviour, the same as ninfer and the Qwen3 template.
- #17855 (merged 2026-08-19, first shipped in v0.32.15) added `normalizeQwen38Messages` (L136-170, called L203-208) for the
  **qwen3.8 variant only**. It collects every `system`/`developer` message anywhere in the history, `TrimSpace`s it, drops empty
  ones, and joins them with **`"\n\n"`** into one leading system message. The other messages keep their order. This is not configurable.
- Go-template path (non-renderer models, custom GGUF imports): `template/template.go` `collate()` L357-374, called from `Execute` L258.
  It merges **consecutive same-role** messages (not `tool`) with `"\n\n"`. For `[sys, sys, user]` that means one system message
  `sys0\n\nhook` inside `{{ range .Messages }}`. Non-adjacent system messages stay in place. `.System` is every system message joined with `"\n\n"`. Whether a template uses `.System` or loops over `.Messages` decides what reaches the model.

## 6. HF transformers `apply_chat_template`: raises.
`src/transformers/utils/chat_template_utils.py` L478-479: `def raise_exception(message): raise jinja2.exceptions.TemplateError(message)`.
It does no pre-processing, so the caller gets the exception. Every Python server above catches that exception and decides the status itself.

## 7. Comparison

| Engine (2026-09) | `[sys, sys, user]` with Qwen3.5/3.6/3.8 template | Separator | Status on failure | Configurable |
|---|---|---|---|---|
| HF transformers | raises `TemplateError` | — | exception | edit template |
| vLLM | rejects | — | **400** BadRequestError | `--chat-template`, `--trust-request-chat-template` |
| SGLang | rejects | — | **400** | `--chat-template` |
| llama.cpp `--jinja` | rejects | — | **500** now (400 in older autoparser builds) | `--chat-template-file` |
| llama.cpp legacy ChatML | separate block in place | own `<|im_end|>\n<|im_start|>system\n` | — | — |
| Ollama qwen3.5/3.6 renderer | separate block in place | same as above | — | no |
| Ollama qwen3.8 renderer | merged into leading system, moved to front | `"\n\n"`, trimmed, empties dropped | — (500 before 0.32.14) | no |
| Ollama Go-template path | adjacent system messages merged; non-adjacent ones in place | `"\n\n"` | — | edit Modelfile TEMPLATE |
| Qwen3 / Qwen3-Next official jinja | separate block in place | same as above | — | — |
| **ninfer** | separate block in place | same as above | — | — |

No engine applies the template's rule as written *and* merges on the server's side by default, except Ollama for qwen3.8. vLLM's merge PRs were not merged.

## 8. Prefix-caching implications (second system line changes between requests)

- **Separate block in place** (ninfer, Ollama 3.5/3.6, Qwen3 template, legacy ChatML). The tokens up to the end of block 0's
  `<|im_end|>\n` (tools block + big agent prompt) do not depend on the hook line. When the hook line changes, only the cache from its own
  block onward is invalidated. For `[sys0, sysHook, user...]` that is almost everything after the big prompt, which is the best
  possible outcome anyway. A system line injected mid-conversation invalidates only from its position, so the earlier history stays cached.
- **Merge into the leading block, appended** (Ollama qwen3.8, Ollama Go-template collate, vLLM PR #44505 design). `sys0 + "\n\n" + hook` is
  **not** fully token-exact for sys0. The Qwen pre-tokenizer regex has ` ?[^\s\p{L}\p{N}]+[\r\n]*`, so trailing punctuation absorbs the newlines.
  Measured with the Qwen3-8B `tokenizer.json`: `"...rules."` → `[..,'Ġrules','.']`, but `"...rules.\n\nHook: x"` → `[..,'Ġrules','.ĊĊ','Hook',...]`.
  Standalone, `"rules.<|im_end|>"` keeps `'.'`. The last sys0 token therefore differs between the merged and separate forms. Across requests
  it stays stable as long as a hook is always present, whatever its text. The trim also makes whitespace-only hook changes re-render identically.
  The real cost is mid-conversation system messages: merging *moves* them to the front, inside the first block. Each new or changed
  injected line then rewrites the prompt from the end of sys0, which **invalidates the whole conversation history's KV** on every such turn.
  With tools present, the Qwen3.5+ template also puts system content inside the tools system block after `</IMPORTANT>\n\n`. The tools
  JSON stays a stable prefix, but the block's `<|im_end|>` moves.
- **Reject** (vLLM, SGLang, llama.cpp jinja). The server renders nothing. Clients or proxies usually respond by concatenating all system text into
  `messages[0]` (opencode#16560, open-webui#22505). That has the same cache behaviour as merging, and it hurts the most when the system text sits at the front.
- **Merge into the next user message** (llama.cpp `system_message_not_supported`, templates without a system role only). The hook text lands at the
  head of the user turn with `"\n"`. The prefix is stable up to that point. It does not apply to Qwen.

Conclusion: of these behaviours, only ninfer's in-place block keeps every token before the changed system line exact, wherever
that line sits in the conversation. The front-merge behaviours are prefix-stable only when every system message is already leading.

Sources: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/chat_template.jinja · https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja ·
https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json · https://github.com/vllm-project/vllm/blob/a615f53364c109ade2b6bb69248e16ed40405d1f/vllm/renderers/hf.py ·
https://github.com/vllm-project/vllm/issues/41114 · /pull/44505 · /pull/44643 · /pull/46976 ·
https://github.com/ggml-org/llama.cpp/blob/4bc272fd729bd094c0422e4b8353da8d2fec91f8/common/chat.cpp · https://github.com/ggml-org/llama.cpp/issues/27367 · /issues/20733 · /pull/27131 ·
https://github.com/sgl-project/sglang/blob/408d2334c34d387a36a26398dff9a8f004328344/python/sglang/srt/entrypoints/openai/serving_chat.py ·
https://github.com/ollama/ollama/blob/6b8bef2caa5bd46b50e13a9dd893fcedc9f3aaea/model/renderers/qwen35.go · https://github.com/ollama/ollama/pull/17757 · /pull/17855 · /issues/17774 ·
https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py · https://github.com/anomalyco/opencode/issues/16560
