---
title: "Settings"
description: "The chat's settings panel: the system prompt, thinking and its budget, sampling, the lane tag, parallel sessions, the transport and the tools."
sidebar:
  order: 3
---

The right-hand column of the chat holds what the next message is sent with.
Nothing here is sent until you send, and nothing is stored: a reload puts the
defaults back, apart from the three settings marked *stored* below. It has two
tabs, **General** and **Tools**.

![The General tab](../images/playground/settings-general.png) ![The Tools tab](../images/playground/settings-tools.png)

## General

| Control | Default | What the request carries |
|---|---|---|
| **System prompt** (`User` tab) | empty | your text, after the ignis prompt |
| **System prompt** (`ignis` tab) | read-only | what the tools that are on write: the date, how to call agents, the web, and so on |
| **Thinking**: Off, Low, Medium, Xhigh, Max | Xhigh | `reasoning_effort`: `none`, `low`, `medium`, `xhigh`, `max` |
| **Thinking budget**: Default, Off, 2K, 4K, 8K, 16K, or a number | Default | `thinking_budget`: nothing for Default, `0` for Off, else the count |
| **Temperature** | 1.00 | `temperature` |
| **top_p** | 0.95 | `top_p` |
| **Max tokens** | 16384 | `max_tokens` (HTTP) or `max_output_tokens` (WebSocket); empty sends none and the server's default applies |
| **Lane tag**: Interactive, Agent | Interactive | `class` |
| **Parallel sessions** (*stored*) | off | nothing: page behaviour |
| **Dictation stops after**: Off, 2 s, 3 s, 5 s (*stored*) | 3 s | nothing: the silence the mic's dictation asks the server to end on |
| **Transport**: WebSocket, HTTP (*stored*) | WebSocket | the route the chat uses |
| **Keeping the sessions** | read from the browser | nothing: asks the browser to keep this page's storage |
| **Markdown in replies** | on | nothing: display only |

The system message the server receives is the ignis prompt (what the enabled
tools add), a blank line, then yours. It is sent once per session and stays
the same from turn to turn, so the server can reuse the whole prompt prefix
([prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)).

### Thinking and its budget

**Thinking** is how hard the model reasons before it answers; **Off** answers
straight away, **Max** reasons without a budget. The **thinking budget** caps
how many reasoning tokens a reply may spend: past it, the server closes the
reasoning with a sentence that tells the model to answer from what it has
([reasoning](../concepts/reasoning.md)). **Default** is the server's own
(`model.thinking_budget`, 32,768), **Off** is no budget. With Thinking Off the
budget is inert and greyed out; with Max it is ignored. A budget larger than
Max tokens is shortened by the server to leave the answer room, and the panel
says so. When the budget closes a reply, it carries an orange **Budget
reached** mark and the session log reads `reached`.

### Sampling

Temperature and top_p go to the server as they are. Temperature 0 needs top_p
at 1 (greedy decoding), or the server answers 400; the panel warns in red
before you send.

### Lane tag

**Interactive** marks the conversation; **Agent** marks background work that
yields first when the engine is short of memory, the class subagents run
with. What the tag changes on the server:
[Lanes that know who is asking](../why-ignis/lanes.md).

### Dictation

**Dictation stops after** is how long a silence ends a dictation by itself, as
if you had pressed Done ([dictating a prompt](chat.md#dictating-a-prompt)). It
is the same for every mic on the page: the chat's, the Image studio's and
Decide's. A shorter pause cuts nothing, and a held Ctrl+Shift+Space (push to
talk) never stops by itself. **Off** leaves the end to you: Done, a second tap
of the shortcut, or letting go of a held one. The choice is kept by this
browser, not the server.

### Parallel sessions and the transport

Off, the page runs one turn at a time across all sessions. On, you can send
in one session while another still streams (a session still answers one
reply at a time). The **WebSocket** carries every stream at once and sends
only what is new each turn; **HTTP** opens one connection per stream, and the
browser allows five, so the rest wait. The Decide tab uses HTTP either way.

### Keeping the sessions

The sessions live in this browser's storage, which a browser treats as
best-effort: when the disk runs short it may delete an origin's data whole.
This section reads the state each time Settings shows and says one of three
things: **Kept** (the browser will not delete the sessions on its own),
**Best-effort** (it may, when the disk runs short), or that this browser has no
Storage API, which leaves nothing to ask.

When the storage is best-effort, **Ask the browser to keep them** asks on your
click. Firefox puts the question to you. A Chromium-based browser answers by
itself, and refuses a site it does not count as important. The section then
says it was refused and, while the notifications permission can still be asked
for, offers **Allow notifications, then ask again**: a Chromium-based browser
counts that permission as one sign that a site matters. The Playground sends no
notifications; the permission is only that sign. That click asks the browser
for the permission, then asks again to keep the storage, and the section says
what came of it. If the browser still refuses, installing the page as an app
(the browser's install option) is the other way. The button is not offered when
the browser has no Notification API or the permission is already allowed or
blocked, and the section says so. Nothing here is asked until you click, apart
from the one request the Playground makes at start
([sessions](sessions.md#asking-the-browser-to-keep-it)).

## Tools

**All tools** turns the tool set on and off as a whole. While it is on, **Tool
rounds** (16) caps how many replies in a row may call tools, in a turn and
inside each agent: past it, the calls come back unrun. Then one switch per
tool, each explained in [Tools and agents](tools-and-agents.md):

| Tool | Default | Extra control |
|---|---|---|
| Agents | on | |
| Web | on | a Tavily API key, kept in this browser |
| Ask me | on | |
| Date and time | on | **Update every prompt** |
| Run JavaScript | on | **Safety check** (on) |
| Plan | on | |
| Memory | on | **Open memory** |
| Workspace | on | |
| Pictures | on | |

A tool is declared to the model only while its switch is on, and its
instructions go into the ignis system prompt; the `ignis` tab of the system
prompt shows the text.

## Read more

- [Prompting](../configuration/prompting.md): the server-side defaults behind Default.
- [Chat completions](../api/chat-completions.md): every request field, including the ones the panel does not set (`top_k`, `seed`, `stop`, penalties).
