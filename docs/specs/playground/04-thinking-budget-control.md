# playground 04 — Let the user choose the thinking budget

GitHub: #266

## Problem Statement

The server can now cap how long a request reasons before it forces the model's own
close. There is a per-request `thinking_budget` and a server default,
`--thinking-budget` (spec server/04 §"Thinking budget"; default and opt-out in spec
server/08). The Playground cannot use either:

- its settings choose the reasoning effort, temperature, top-p and `max_tokens`, but
  never send `thinking_budget`;
- every Playground turn and every agent the Agents tool spawns runs at whatever the
  server default is, and a user who wants a quick answer, or an unbounded deep one,
  has no control;
- when the budget does close a reply's reasoning, nothing on screen says so.

## Solution

The settings panel gains a **Thinking budget** control beside the reasoning effort:
server default, off, or a number of tokens. It rides every request the Playground
sends, including the Agents tool's sub-requests. A reply whose reasoning the budget
closed says so on the reply and in the request log row.

## User Stories

1. As a Playground user, I want a thinking-budget control next to the reasoning effort, so that I set how long the model may think in the same place I set how hard.
2. As a Playground user, I want a "server default" choice that sends nothing, so that I get whatever the operator configured.
3. As a Playground user, I want an "off" choice, so that a hard question can reason without a forced close.
4. As a Playground user, I want to type a number of tokens, so that I can pick a tight or a generous budget.
5. As a Playground user, I want a few presets (e.g. 2K, 4K, 8K, 16K), so that I do not have to type common values.
6. As a Playground user, I want the control disabled when thinking is off (effort `none`), so that I am not offered a setting that does nothing.
7. As a Playground user, I want a budget larger than my `max_tokens` flagged, so that I understand the answer-room clamp will shorten it.
8. As a Playground user, I want my choice remembered like my other settings, so that I do not re-enter it every session.
9. As a Playground user, I want parallel sessions to each keep their own choice when settings are per session, so that two experiments do not interfere.
10. As a Playground user, I want the Agents tool's sub-agents to use the same budget as the turn that spawned them, so that a fan-out does not silently think unbounded.
11. As a Playground user, I want a reply whose reasoning was closed by the budget to show a small "budget reached" mark on its reasoning block, so that I know why it stopped thinking.
12. As a Playground user, I want the request log row to show the budget sent and whether it was hit, so that I can compare turns.
13. As a Playground user, I want invalid input (0 typed as a number, negatives, text) rejected in the field, so that I never send a request the server refuses with a 400.
14. As a Playground developer, I want the mock server (`dev:mock`) to accept and echo the field, so that UI work does not need the GPU.
15. As a Playground user on a server without budget support, I want the control to still work harmlessly (the field ignored), so that the Playground works against older servers.

## Implementation Decisions

- **Settings.** A new setting holds one of: *server default* (the field is omitted),
  *off* (`thinking_budget: 0`, per server/08), or a positive token count.
  - The default is *server default*.
  - It is persisted with the other settings, under the same storage rules.
- **Request building.** The request builder adds `thinking_budget` to every chat
  request it builds, the Agents tool's included. The field is omitted when thinking is
  off or the setting is *server default*.
- **Display.** The reasoning block renders a "budget reached" marker when the response
  carries server/08's forced-close extension field. The request log row gains the
  budget sent and the forced flag.
- **No server work here.** The field, `0` = off, and the forced-close field are all
  server/08's. This spec consumes them.

## Testing Decisions

- A good test renders the component or calls the request builder and asserts what a
  user sees or what goes on the wire. It never tests internal state.
- **Request builder unit tests:** prior art `web/src/api/request.test.ts`, the
  `reasoning_effort` cases. Cover:
  - default → field absent;
  - off → `0`;
  - number → that number;
  - thinking off → absent;
  - Agents tool sub-request carries the same value (prior art
    `web/src/tools/agents/agents.test.ts`).
- **Settings panel test** (prior art `SettingsPanel.test.tsx`): presets, typed value,
  validation, disabled when effort is `none`, the `max_tokens` warning.
- **Reply rendering test:** the marker appears only when the response's forced field is
  true.
- **Mock server:** the mock accepts the field and can be told to report a forced close.

## Out of Scope

- Any server-side behaviour: default value, clamp, telemetry (server/08).
- A budget per message within one conversation.
- Showing a live reasoning-token counter while streaming.

## Further Notes

- Depends on server/08 for the `0` opt-out and the forced-close field. Against a
  server without them, the control's number presets still work; "off" and the marker
  do not.
- The Playground's brand tokens and component conventions (`web/src/brand`, `web/src/ui`) apply.
