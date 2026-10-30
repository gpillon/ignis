---
title: "Sessions and storage"
description: "The chat's sessions, what the Playground keeps in this browser, and what a reload forgets."
sidebar:
  order: 5
---

A **session** is one conversation. The left-hand column lists them:
**New session** starts one, a click opens one, and each row reads the
first line of its first prompt with how many replies it has (or `Streaming`,
with an orange square, while one streams). The cross on a row deletes it, with
its stream and the pictures no other session still uses.

![Three sessions](../images/playground/sessions.png)

A session answers one reply at a time. Several sessions can stream at once
when **Parallel sessions** is on ([Settings](settings.md#parallel-sessions-and-the-transport)):
over the WebSocket they share one connection; over HTTP the browser carries
five streams, agents included, and the rest wait. Each session is a separate
conversation for the server too: its prompt state is retained and found by
content, so going back to an older session resumes from the cache when the
server still holds it ([State that outlives its request](../why-ignis/state-that-outlives.md)).

## What is kept, and where

Everything the Playground keeps lives in **this browser**, for this origin;
nothing is sent to the server for safekeeping.

| What | Where | Survives a reload |
|---|---|---|
| Chat sessions: every message, its reasoning, tool runs, figures and log, the workspace it is in | IndexedDB `ignis-playground`, collection `chat-sessions` | yes |
| Workspaces: their names, and every file with every version and its author | same database, `workspaces`, and `workspace-files/<id>` for each one's files | yes; deleting a chat keeps its workspace, which goes only when you [delete it](chat.md#the-workspace-picker) |
| Decide's decisions: drafts, evidence, answers | same database, `decide-sessions` | yes |
| Image sessions and their steps | same database, `image-sessions` | yes |
| Transcripts: each utterance's text, times and speaker, the speakers' names and the corrections, the context it is told, the suggestions and what became of them, how far the text model has checked it, the notes, the questions asked and their answers, never audio | same database, `transcribe-sessions` | yes; a listening under way ends, its completed utterances kept |
| Pictures (sent in the chat, made, marked, a workspace's), stored once by their SHA-256 and shared | same database, `images` and `bytes` | yes; deleting a chat, an image session or a workspace frees only the pictures nothing else names |
| A workspace's other bytes, such as a PDF's, stored once | same database, `blobs` | yes |
| The session that was open, per view | same database, `active-session` | yes |
| The API key | `localStorage` `ignis.apiKey` | yes |
| Parallel sessions, the HTTP transport | `localStorage` `ignis.parallelSessions`, `ignis.httpTransport` | yes |
| The workspace panel, open or closed | `localStorage` `ignis.workspacePanel` | yes |
| The Tavily key, memory notes, the image view's "Enhance the prompt" | `localStorage` `ignis.tavilyKey`, `ignis.memory`, `ignis.imageEnhance` | yes |
| The Transcribe view's file pace, its silence, its Hold my mic, its notes' kind and Keep updating, Suggest while listening | `localStorage` `ignis.transcribeFilePace`, `ignis.transcribeSilence`, `ignis.transcribeHoldMic`, `ignis.transcribeNoteKind`, `ignis.transcribeNotesLive`, `ignis.transcribeSuggestions` | yes |
| Settings, tool switches, the prompt box, the Server view's draft, the Monitor's history | memory only | no |

A reply that was streaming when the page closed comes back stopped, with the
text it had got; a tool call it left running is answered as stopped.

:::note[One store per origin]
The page served by ignis at `http://127.0.0.1:8000/ui/` and the development
server on `localhost:5173` are different origins, so they keep separate
stores. Clearing the site's data in the browser deletes the sessions.
:::

When the browser refuses the space, the session list says so in red and the
page goes on without saving. The same happens when the Playground's store has
moved to a newer version (workspaces brought one) while another tab of the
same origin still runs the older page: the list says another tab holds the
store at an older version, and once that tab is closed a reload of this one
reads the sessions back.

## Asking the browser to keep it

A browser treats this storage as best-effort: when the disk runs short it may
evict an origin's data whole, with no prompt and no trace. So at start the
Playground asks the browser to make its storage **persistent**, once per page
load and only when it is not persistent yet. A browser that grants it no
longer evicts the sessions on its own; only clearing the site's data deletes
them.

A browser may refuse, and a refusal changes nothing else: the page opens and
saves as before, but the storage stays best-effort, and a short disk can still
take the sessions. Firefox asks you. A Chromium-based browser decides without
asking, and keeps only the storage of sites it counts as important, from
signals such as use, installation and the permission to show notifications;
bookmarking the page was measured not to be enough. So the Playground can ask
on purpose: **Settings**, **General**, **Keeping the sessions** says whether
this browser keeps them and has a button that asks again, then, if the browser
refuses, a second one that asks for the notifications permission and asks
again. See [Keeping the sessions](settings.md#keeping-the-sessions).

To see the state, open the browser's DevTools console and run
`await navigator.storage.persisted()`: `true` means this origin is kept. In a
Chromium-based browser's DevTools the **Application** panel, **Storage**
section, shows how much the origin uses against its quota; in Firefox's, the
**Storage** panel lists what it keeps.

## Read more

- [Chat](chat.md): the session log at the foot of each session.
- [KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md): what the server keeps of a conversation, and for how long.
