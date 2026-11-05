---
title: "Server"
description: "The Server view: what each engine runs, the models a switch can go to or the server can fetch, and the running configuration as a form you can apply."
sidebar:
  order: 10
---

The **Server** view changes the running server from the page. At the top, the
**status bar**: what each engine runs now (the main model, the speech model,
the diarizer) and how the last switch ended. It stays in place while the two
tabs under it scroll: **Models**, every model the server can switch to or
fetch, and **Configuration**, every field the server runs with, as a form.
Applying a change sends one `PATCH /v1/config`; the server judges every value,
applies what it can live, and reloads the model only for a field that needs
it. A change of tab leaves the other as it was: a draft, a search and a scope
stay.

![The status bar and the Models tab](../images/playground/server.png)

<div class="ignis-video ignis-tour not-content">
  <iframe src="https://www.youtube-nocookie.com/embed/xnyLYSv4DFI" title="The ignis Server view: switch models, download and configure, without a restart" loading="lazy" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

Panel tour 5 on YouTube: [The ignis Server view: switch models, download and configure, without a restart](https://www.youtube.com/watch?v=xnyLYSv4DFI).

## The status bar

The headline is the loaded model's id with its **State** (`Serving`,
`Switching`, `Loading`, `Unloaded`, `Not serving`), its **Context** in tokens
and its **Family**. Beside them, the models loaded beside it rather than
switched to: **Speech**, the
[speech model](../api/realtime-transcription.md)'s id, and **Diarizer**, the
model that gives a live transcript its
[speakers](../api/realtime-transcription.md#speakers), with whether it is
**Loaded** (the first session that asks for speakers loads it, the speech
model's idle unload frees it) or **Not loaded**. Either one that cannot serve
reads **Unavailable**, and **Why** beside it opens the server's reason under
the figures: turned off, or its files missing, with the
`ignis-server model download` command that fetches them.

During a switch the bar says what is happening and for how long; afterwards
it keeps the outcome: `Switched from … to …`, or why the switch was refused
or failed and which model serves now. A model the server
[unloaded while idle](../api/idle-unload.md) shows as **Unloaded**, not as an
error; the next message loads it again, and **Load** does it now. While a chat
reply streams, the bar says how long a switch would let it finish
(`switch.drain_timeout`) before cancelling it.

![A switch from the 27B to Flash-Next, under way](../images/playground/server-switching.png)

## Models

The **Models** tab lists every model the server's catalog knows
(`GET /v1/models/catalog`), under a line that totals what they take on disk and
the room left on the volume of the server's `download.path`:
`The listed models take <size>. <size> free under <path>.`, without the second
sentence when the server could not read it. The rows are grouped by the engine
they are for, and a group with no model is left out:

- **Main model**: the text and image models, one loaded at a time; **Load**
  switches the server to one.
- **Speech**: the models that transcribe. No row loads one: `speech.model`, in
  Configuration, picks it.
- **Diarizer**: the models that tell speakers apart, picked by
  `diarizer.model` the same way.

Each row says its **Type** (`Text`, `Image`, `Speech` or `Diarizer`, known
before a built-in entry is downloaded, and blank only when nothing tells it,
as for an operator entry of no family not yet downloaded), its **Source** (the
built-in catalog, an [operator catalog](../guides/operator-catalog.md), or a
known model), how much of it is **On disk**, and its action. Under a catalog
model's id sits its **license**, the identifier (`apache-2.0`,
`qwen-community-1.0`...) as a link to the license's text, from the entry's
`license` ([the catalog](../guides/models.md#the-catalog)); an entry that
states none shows none. While a download is sent or runs, its row says the
same again: downloading it, you are bound by its license, with the link. The
page asks for no acceptance: the terms bind whoever downloads, as with any
download from the hub.

- **Loaded** for the model that serves;
- **Load** for a main model on disk: it switches to it by id
  (`POST /v1/models/switch`), whatever `switch.allow_implicit` says;
- **In use** for the speech model or the diarizer the configuration picks,
  while it can serve; another one on disk offers only **Remove**;
- **Download** for a catalog entry not on disk: the server downloads it
  ([Downloading a model](#downloading-a-model));
- **Resume** for one partly downloaded, a cancelled download's included: the
  server downloads the rest ([Cancelling and resuming](#cancelling-and-resuming));
- **Cancel** while its download runs: it stops it, keeping what it fetched;
- **Not for this card** for an entry above the card's class or its GPU
  generation ([the catalog](../guides/models.md#the-catalog)), with the
  server's reason as its tooltip: neither loaded nor downloaded;
- **Copy download command** in place of **Download** or **Resume** when the server does not
  download from the page (`download.enabled` or `download.from_ui` false): it
  copies `ignis-server model download <id>`;
- **Remove** beside **Load** on a catalog entry on disk that is not loaded, and
  beside **Resume** on one partly downloaded: it deletes the entry's files from
  the server, after a confirmation ([Removing a model](#removing-a-model)).

A refused **Load**, **Download** or **Remove** shows the server's reason above
the list, where the outcome of a removal shows too.

## Downloading a model

**Download** sends `POST /v1/models/download` with the entry's id, and the
server fetches it into its own `download.path`, whatever machine the browser
runs on, with the same transfer as `ignis-server model download`: each file
resumed from its `.part`, checked against its pinned size and SHA-256
([Get a model](../guides/models.md#downloading-from-the-server-view-post-v1modelsdownload)).
It runs in the background while the loaded model keeps serving. The row then
reads `Downloading, <on disk> of <total>, <speed>/s`: the page reads
`GET /v1/models/catalog` every second while a download runs, and the speed is
the bytes landed between those reads. One download runs at a time, so every
other **Download** and **Resume** is out of reach until it ends; so is every one during a
switch, or while no model serves.

Once every file verified, the row reads `On disk` and offers **Load**: the
server never switches to a model by itself. A partly downloaded entry offers
**Resume** instead. A download that fails keeps its reason under the row,
`Download failed: …`, until the entry's next download or a restart, or until
the entry is on disk whole by another way (`ignis-server model download`): a row
that reads `On disk` shows no failure. A gated
repository fails naming the token the server reads from its environment
(`IGNIS_DOWNLOAD_TOKEN`, or `HF_TOKEN` for huggingface.co). The page never asks
for a token nor shows one. A refused **Download** (another one running, the
entry already on disk, downloading turned off) shows the server's reason above
the list.

Before a byte is written, the server checks that the volume of its
`download.path` has room for the entry's bytes still missing plus
`download.free_margin` (`1G` by default). Short of room, **Download** (or a
partial entry's **Resume**) is refused, and the reason, naming the bytes needed
and the bytes free, shows under
the row rather than above the list, `Download failed: not enough room for …`,
until the entry's next download, a restart, or the entry on disk whole
([Room on the disk](../guides/models.md#room-on-the-disk---download-free-margin)).
Free some space there and press it again. `download.free_margin` is
shown read-only, `Set at start`, in the Download group.

An entry that `ignis-server model download` is fetching into the server's
`download.path` is refused too, with `409 download_locked` naming the command
and its process id, until the command ends: each fetch of an entry holds its
lock, the server's and the command's
alike, so two never write the same file
([one writer per entry](../guides/models.md#one-writer-per-entry)). Its row
shows no download of the server's, and any other entry downloads meanwhile.

Downloading takes the same API key as **Apply**
([ADR 0071](../../adr/0071-the-server-downloads-and-removes-models-from-the-server-view.md)).
`download.from_ui` false turns it off for the page and the API, and
`download.enabled` false keeps the server off the network altogether; both are
shown read-only, `Set at start`, in the Download group.

### Cancelling and resuming

**Cancel** on the running row sends `POST /v1/models/download/cancel`, and the
server stops the download before it answers: the files it verified stay at
their names, and the file it was writing stays as its `.part`, with every byte
that arrived. The row then reads `Partly downloaded, <on disk> of <total>` and
offers **Resume**, which sends `POST /v1/models/download` again: the server
fetches only what is missing, from the `.part` on, and checks each file against
its pinned size and SHA-256 before it renames it into place, as for any
download ([Get a model](../guides/models.md#cancelling-and-resuming-post-v1modelsdownloadcancel)).
**Cancel** stays in reach during a switch, since stopping a download never
touches the loaded model; it is out of reach only while it is being sent or
while the server is not answering, and takes the same API key as **Download**.

A download cut short by a dropped connection or a restart of the server leaves
a partial entry too, shown the same way. Nothing resumes by itself: after a
restart the entry stays partial until **Resume**, or `ignis-server model
download <id>` on the server's machine, continues it. The one exception is the
model the server is started to serve, which a start fetches when it is missing,
as before ([Letting the server fetch it](../guides/models.md#letting-the-server-fetch-it)).

## Removing a model

**Remove** asks first, under the row: `Remove <id>? Its files under
download.path go, <size>; it can be downloaded again.`, with `its unfinished
download included` on a partly downloaded entry. When the server started on
that entry's files, the confirmation also says what the next start does without
it: it downloads it again, asking first on a terminal, or, with
`download.enabled` false, comes up with no model, on the placeholder template; a
start whose `model.artifact` names its file refuses to start. When
`switch.known_models` names the entry's file, it says a switch to that model is
refused until it is downloaded again. Either is named by the id it is served
under when that is not the entry's own (`The server started on it (as boot)`).
**Remove** there sends `POST /v1/models/remove` with the id; **Keep it** sends
nothing.

The server deletes the entry's files under its `download.path`, each at its
name and its `.part`, and answers once they are gone
([Get a model](../guides/models.md#removing-a-model-post-v1modelsremove)). The
line above the list then says `Removed <id>: <size> freed.`, and the row reads
`Not downloaded` and offers **Download**. A file another entry also lists stays
while that entry is on disk or being downloaded, and that line names it,
`Kept <file>, which <entry> also lists.`: the entry removed then reads partly
downloaded.

The server refuses, with its reason above the list, the model loaded (also when it
was unloaded while idle, or serves from the entry's files under another id),
the model a switch or a load is loading, the speech model the speech worker has
loaded (until it unloads it, `speech.idle_unload` seconds after its last
transcription), and an entry being downloaded, by the server (**Cancel** it
first) or by `ignis-server model download` on its machine. The page offers no **Remove** on the loaded row, the one loading or
the one downloading, and **Remove** is out of reach while a removal is sent,
during a switch, or while no model serves. A known model that is no catalog
entry is a path the operator gave, and is never removed.

Removing takes the same API key as **Apply**
([ADR 0071](../../adr/0071-the-server-downloads-and-removes-models-from-the-server-view.md)).
`download.from_ui` false turns it off with downloading, and the page then
offers no **Remove**. `download.enabled` false does not: a server kept off the
network still frees its disk.

## The configuration

The **Configuration** tab holds the form; an orange count beside its name says
how many changes the draft holds, whichever tab is open. **Scope** picks what
the form shows and edits. **All models**, which the form
opens on, shows every field with its shared value. A model family shows only
what that family can set for itself: the fields that take a value per family
(its own, which wins over the shared one) and the fields only that family
takes; a field with one value for every model, such as
`server.request_timeout`, is changed in All models. The running family's
segment is filled orange. The groups
(Server, Model, VRAM, Prompt reuse, Model switch, Speculative decoding,
Vision, Media, image, Download, N-gram table, KV-disk) are listed on the left;
an orange count marks the groups holding a change.

**Find a field**, beside the scope, narrows the form to the fields that hold
every word typed, case aside, in their key, their description, their flag or
their environment variable: `max context`, `--spec-backend` and `IGNIS_VRAM`
each find theirs. The part of a name it matched is marked; a field found by its
flag or variable shows them under its name, and one found in the rest of its
description opens it. A group with nothing found is dimmed on the left. The
search only hides rows: a change on a hidden one stays in the draft and is
applied with the rest. **Escape** clears it.

![The Model group: values, their source, and the fields that reload](../images/playground/server-config.png)

Each row is one field ([every field](../reference/configuration.md)): its
name, the first sentence of its description (**More** for the rest), and
where its value comes from: `From profile rtx5090`, `Set by
IGNIS_MODEL_THINKING_BUDGET` (a flag or an environment variable), `Decided by
the card, NVIDIA GeForce RTX 3090 (8.6)` (the prompt attention's integer band
and the dense prefill on a card below compute capability 12.0), `Shared`, or `Own value`. A **reload** mark says a change to it reloads the model. A
field set only at start (the bind address, the API key, the artifact) is shown
read-only, `Set at start`.

Editing fills a draft and sends nothing. The dock at the foot counts the
changes and offers **Discard** and **Apply**, which reads **Apply and reload
&lt;model&gt;** when the draft holds a field that reloads it. The label is the
confirmation; there is no dialog.

- A live change answers at once: `Applied and written to <path>`, the config
  file in use, or why it was not written. A value a flag or an environment
  variable would override at the next start is named.
- A reload is followed in the status bar, then `Reloaded <model> and written
  to <path>`.
- A refused value is named by the server; its row is marked `Refused by the
  server`, and nothing was applied.

**View as config file** shows the configuration in force, live changes
included, as `ignis.yaml`, to copy or download.

## Read more

- [Configuration](../configuration/index.md): fields, precedence and profiles.
- [Models that come and go](../why-ignis/models-that-come-and-go.md): what a switch does and costs.
- [Model switching](../api/model-switching.md): the same switch from your own code.
