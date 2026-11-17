---
title: "The connector: ignis's tools in a coding agent's harness"
description: "Add ignis to opencode, qwen-code, pi or DeepSeek Harness as one MCP server at /v1/mcp: the server tools the agent sees, how a result and an error reach it, and the two settings."
sidebar:
  order: 6
---

From 0.11.0, ignis serves an MCP server of its own at `/v1/mcp`, on the port
and behind the key the rest of `/v1` has. A coding agent's harness adds it as
one MCP server entry, and the model in the harness sees ignis's **server
tools**: what the server can do beyond chat, as tools it can call. With ignis
as the harness's model too, the agent works on the same server it calls.

## The MCP server

- **The URL** is the base URL's `/v1/mcp`: `http://127.0.0.1:8000/v1/mcp` on
  the default bind, the server's LAN address or its tunnel's on a server
  reached from elsewhere ([Access](../configuration/access.md)). Any `Host` is
  admitted, as on every other route.
- **The key**, when the server has one, goes in the entry as a header,
  `Authorization: Bearer <key>`: the same key as the rest of `/v1`. Without
  it the route answers `401 invalid_api_key`.
- **Streamable HTTP, stateless.** Every request stands alone, with no session
  id, so a restart of ignis leaves no harness holding a session that is gone.
  An answer is `application/json`, or an event stream when the call sends
  progress first. There is no stream from the server (`GET` is `405`).
- **It answers through a model switch** and while a model warms up:
  `initialize` and the tool list never wait. A tool's call goes to its route
  and is held, or refused, as a request of the client's own would be.
- **The tool list is fixed at start.** It is the same bytes whatever model is
  loaded, before and after a switch or a configuration change, so it stays in
  a harness's prompt prefix and [prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)
  keeps working. A server started with other connector settings lists other
  tools.
- **`instructions`**, which `initialize` answers and most harnesses put in the
  model's prompt, are fixed at start too: under 1 KB, the first line saying
  what ignis is.

## Each harness's entry

The entries below are read from each harness's own source at the version
named; none has been run against ignis yet. Every harness asks for the MCP
server's URL and, with a key, the `Authorization` header.

The key is read from the harness's environment, as `IGNIS_API_KEY` in the
entries below: set it to the server's key in the shell the harness starts
from. The name is the harness's to choose: the provider example on
[Connect a client](../start/connect-a-client.md#coding-agents-opencode-and-others)
reads the key from `IGNIS_SERVER_API_KEY`, the server's own variable, and
either works where it holds the key. On a server with no key, leave the
header out.

### opencode

opencode 2.3.0 reads its MCP servers from `opencode.json` (or
`opencode.jsonc`): `~/.config/opencode/` for every project, the project's
root or its `.opencode/` folder for one.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "ignis": {
        "type": "remote",
        "url": "http://127.0.0.1:8000/v1/mcp",
        "oauth": false,
        "codemode": false,
        "headers": { "Authorization": "Bearer {env:IGNIS_API_KEY}" }
      }
    }
  }
}
```

- `oauth: false`: opencode tries OAuth on a remote server unless told the
  header is the credential.
- `codemode: false` declares the tools to the model as tools. With opencode's
  default, Code Mode, the model reaches them only from code it writes.
- `{env:IGNIS_API_KEY}` is opencode's own spelling; a variable that is not
  set becomes an empty key, which the server answers `401`.
- opencode 1.x takes the server one level up, `"mcp": { "ignis": { … } }`,
  without `codemode`; 2.x still reads that form.
- opencode asks before every MCP call, whatever the tool's hints say.

### qwen-code

qwen-code 0.27.4 reads `mcpServers` from `~/.qwen/settings.json`, or from the
project's `.qwen/settings.json`, whose servers wait for your approval.

```json
{
  "mcpServers": {
    "ignis": {
      "httpUrl": "http://127.0.0.1:8000/v1/mcp",
      "headers": { "Authorization": "Bearer $IGNIS_API_KEY" },
      "alwaysLoadTools": true
    }
  }
}
```

- `httpUrl` is Streamable HTTP; `url` would be SSE, which ignis does not
  serve.
- `alwaysLoadTools: true` puts the tools in the tool list the model gets with
  every request. Without it qwen-code defers them behind its tool search, and
  the list in the prompt prefix changes as the agent finds them.
- `"trust": true` stops qwen-code asking before each call, in a trusted
  folder.
- A variable that is not set stays in the header as written
  (`Bearer $IGNIS_API_KEY`), which the server answers `401`.

### pi

pi 1.5.2 reads `~/.pi/agent/mcp.json`, or `.pi/mcp.json` in a trusted
project. The command

```sh
pi mcp add ignis --url http://127.0.0.1:8000/v1/mcp --bearer-token-env-var IGNIS_API_KEY --exposure direct
```

writes this entry:

```json
{
  "mcpServers": {
    "ignis": {
      "url": "http://127.0.0.1:8000/v1/mcp",
      "headers": { "Authorization": "Bearer ${IGNIS_API_KEY}" },
      "exposure": "direct"
    }
  }
}
```

- `exposure: "direct"` declares the tools to the model as tools. With pi's
  default, `codemode`, the model reaches them only from code it writes, and
  sees only the first line of `instructions`.
- A variable that is not set fails the server's connection, naming it.
- A call is cut after `timeout` seconds, 60 by default, unless it sends
  progress.

### DeepSeek Harness, best effort

DeepSeek Harness dsh-v0.2.2-rc.3 has no MCP settings page: a server is a row
in a YAML patch, `$DSH_HOME/profiles/<profile>/cordis.patch.yml`
(`DSH_HOME` is `~/.dsh` by default). Add the row to the file's `insert` list
rather than replacing the file:

```yaml
- insert:
    - id: mcp-ignis
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: ignis
        transport: streamable-http
        url: http://127.0.0.1:8000/v1/mcp
        headers:
          Authorization: !!js '`Bearer ${process.env.IGNIS_API_KEY}`'
```

- A call is cut after `toolCallTimeoutMs`, 60,000 ms by default, and nothing
  extends it.
- It shows a result's text only: what the connector answers is all text,
  but for `generate_image`'s preview, which reaches the model only when its
  configuration declares image input.
- It is not part of the connector's gate: the entry is read from its source,
  and its own guide may say more.

## The tools

`tools/list` names each tool with a short description the model reads, and
two hints: `readOnlyHint` on the tools that only read, `destructiveHint` on
those whose change cannot be taken back by calling again. Hints are advice to
the harness: opencode asks before every MCP call whatever they say, qwen-code
runs read-only tools in parallel, pi passes them on, DeepSeek Harness reads
none.

### Decisions: `decide`

Asks the model typed questions about some content and answers each one read
off the model rather than generated: a yes/no as the probability of yes, one
option of a set, a level of a score, a number. It calls
[`POST /v1/decide`](../api/decide.md) and answers its body as text. It takes
the route's `state`, the content, and `questions`, keyed by ids the agent
chooses, in the route's own shapes, and nothing else: no `model`, so a call
never switches the server, and no thinking field. Any other argument is a
tool error naming it, before the route is asked. It is not read-only.

The route reads an object's keys in the order they were written, a `state`
record's and a `choice`'s options among them, and so does the tool: the
connector sends the route the arguments exactly as the harness wrote them,
where an MCP library would hand them on sorted by key.

**The 27B only.** Qwen3.8-Flash-Next has no readouts: on a Flash-Next load
the tool's error is the server's own `400 model_unsupported`. The
connector's `instructions` say so in their line on `decide`.

### The catalog: `read_catalog`

The models the server can switch to or download, one row each: whether it is
on disk, which one is loaded, its kind and size, and a running download's
progress. It calls [`GET /v1/models/catalog`](models.md#the-catalog), takes no
argument, and is read-only.

### The configuration: `read_config` and `describe_config`

`read_config` is the configuration the server runs with, as the YAML a config
file holds, live changes included. It calls
[`GET /v1/config?format=yaml`](../configuration/index.md), takes no argument,
and is read-only. It is long, so an agent reads it once and asks for the rest
by field.

`describe_config` says what fields mean: each one's key, kind, default, the
rule a value must meet, whether it changes while the server runs and whether a
change reloads the model. It takes `group` (one group's name), `names` (a list
of keys such as `vram.budget_bytes`, as `read_config` nests them), or both, and
needs at least one: a call with neither is a tool error, so an agent never reads
all the fields at once. Its description lists the groups, read from the
configuration's own table when the server starts. It is read-only.

It calls `GET /v1/config/fields`, which takes the same two filters from any
client: `?group=vram`, and `?names=vram.budget_bytes,server.bind` as one comma
list, never a repeated parameter. The fields come back whole and in
declaration order, whatever the order of the list. With both, a field must be
in the group and named. A group that does not exist, or a name that is no
visible field (a hidden field's key counts as unknown), is `400 config_invalid`
naming it, and nothing else is answered.

### Images: `generate_image`

Makes one image from `prompt`, at an optional `size` (one of `image.sizes`,
or `auto`), and shows the agent a preview of it. It calls
[`POST /v1/images/generations`](../api/images.md), naming the image model the
catalog has on disk, so the agent never needs its id. It is not read-only.

The result is the image first, a PNG preview whose longest edge is at most
512 px (an image within that is shown at its own size), then one line of
text: the size made, that this is a preview and is not kept, and that the
skill's `generate-image` script saves an image at full size to a file.

**The cost.** The card holds one model at a time, so the call switches from
the text model to the image model, about 10 s, and the agent's next turn,
which names its own model, switches back by the
[implicit switch](../api/model-switching.md), about 11 s more. No tool
switches back. Meanwhile every other client is answered
`503 model_switching`, and the text model's prompt state is lost: the next
turn prefills the conversation again
([prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)). With
`switch.allow_implicit` off the image route answers `404 model_not_found`,
and the tool passes it on.

**Seeing the preview** needs vision on in the text model the agent runs on
([`vision.enabled`](../configuration/vision.md)). qwen-code and DeepSeek
Harness pass a tool's image to the model only when the harness's
configuration of the model declares image input; otherwise the model reads
the line of text alone.

**With vision off** the tool makes no image and switches nothing: it answers,
once any image call before it has ended, with a line of text naming the
skill's `generate-image` script, which makes the image as a file, and the
skill's install commands, in a POSIX shell form and a PowerShell form. An
image the model cannot see would cost the switch for nothing, and, sent back
with each later turn, would have those turns refused `vision_disabled`. The
check reads the text model the agent returns to: the loaded model when it is
a text model; with the image model loaded, the text model the server switched
away from, which it remembers. It is made once the call's turn has come, so a
second call in the same turn is judged by the agent's own model, not by the
image model, nor by the enhancer that a call before it visits for
`image.prompt_enhancement`. A server started on the image model knows no
text model, and the tool makes the image.

**Several images in a turn.** The connector's image calls run one at a time.
Asked for three variants, an agent that sends three calls at once has the
second and the third wait for the one before to end: the first switches to
the image model, the others find it loaded. The three cost one switch out
and one back, and none is answered `503 model_switching`. With
[`image.prompt_enhancement`](../reference/configuration.md#imageprompt_enhancement)
on (it is off by default), a call made while the image model is loaded
visits the enhancer to rewrite its prompt, two switches more each. A waiting call is
not a retry and has no deadline of its own: it waits as long as the call
before it runs, and the harness's own timeout gives it up. A call given up
while the switch it began runs makes no image, but the switch runs on, and
the next call waits for it to end as it would for a call that had not been
given up. Behind the
connector, the image route's own queue serves every client, one job at a
time with up to [`image.queue_depth`](../reference/configuration.md#imagequeue_depth)
waiting; a full queue comes back as the tool's error, `503 engine_full`
with its `Retry-After`.

**Progress.** A call that carries a progress token hears its phases as MCP
progress notifications, the answer then an event stream: `waiting` while a
call before it runs, then `switching` and `loading` when it switches, then
the job's steps, `encoding`, each `denoising` step and `decoding`. pi and
opencode 1.x reset their 60 s call timeout on progress, and qwen-code its
300 s idle timer; opencode 2.x gives a call 12 h; DeepSeek Harness sends no
token and cuts a call at 60 s whatever. The image job of a call with a token
is held to [`server.stream_timeout`](../reference/configuration.md#serverstream_timeout),
as a streamed request is; one without, to `server.request_timeout`. A
refusal that arrives once the call's progress has begun, such as a full
queue met after the switch, is the tool's error with no `Retry-After`: the
route sent it as an event of a stream it had already answered `200`.

**The vision budget.** The harness keeps each preview in its history and sends
it back every turn. One merged vision token covers 32 x 32 pixels, so a
512 x 512 preview is 256 tokens, and every image of a request together is
held to [`vision.max_tokens`](../reference/configuration.md#visionmax_tokens),
32,768 by default. Past it the request is refused `media_budget_exceeded`. A
session that makes many images raises it, with `PATCH /v1/config` (it
reloads the model) or at start.

**No image model on disk.** The tool answers `404 model_not_found`, naming
the command that fetches it, `ignis-server model download ignis-image-2.1-turbo`,
run where the server runs, and the tool `download_model` when it is listed.
On a card that cannot load the image model, the error says why instead.

**Licence.** The image model's weights are under the **Qwen Research
License: non-commercial use only**, with a "Built with Qwen" notice. Do not
serve the image model in a commercial product
([Images](../api/images.md)).

### Changing the server: `change_config`, `download_model`, `cancel_download`

These three are listed only when the server was started with
`connector.allow_changes` on (from 0.11.0; [the settings](#the-settings)).
Off, the default, they are not in `tools/list` and a call to one is an
unknown-tool error. The connector's `instructions` name them only then.

`change_config` patches the configuration. It takes one free object, the JSON
merge patch `PATCH /v1/config` takes, shaped as `read_config` shows the
configuration: only the fields to change, a `null` removing a key
(`describe_config` says which fields change live). It is not the schema of
every field. The result is the server's answer as text: what applied, whether
it applied at once (`live`) or reloaded the model (`reload`), the
configuration now running, and whether the change was written to the config
file. A field that needs a reload reloads the loaded model with the whole
patch: the model's prompt state is lost, and every other client gets `503`
until it is back. The tool's description tells the model so, and it is marked
`destructiveHint`.

`download_model` takes a catalog `model` id, as `read_catalog` lists them,
calls [`POST /v1/models/download`](models.md) and answers at once: the
download runs in the background, one at a time, and it does not switch to the
model. The agent follows its progress through `read_catalog`.
`cancel_download` calls `POST /v1/models/download/cancel`, marked
`destructiveHint`: it stops the running download and keeps what it fetched, so
downloading the model again resumes. A refusal of either (`download_in_progress`,
`download_disabled`, `download_not_running`) is the tool's error, the server's
own JSON.

**This is not a security boundary.** The opt-in keeps the three tools out of
the agent's list and its instructions, nothing more. An agent with a shell and
the API key can call `PATCH /v1/config` or `POST /v1/models/download` itself.
Turn it on when you trust the agent with the server; what guards the server is
the key ([Access](../configuration/access.md)).

## Results and errors

A tool's result is the route's answer as text in the result's `content`: the
one part of a result every harness shows its model. `generate_image`'s is
its preview and one line of text instead, as its section above says.
Nothing goes in
`structuredContent`, which DeepSeek Harness never shows, and no embedded
resource, which it drops.

A route that refuses comes back as a tool error, `isError: true`, its text
the server's own error JSON, exactly as a client would read it
(`{"error": {"message", "type", "code"}}`), then a second text,
`Retry-After: <seconds>`, when the route sent that header (a refusal that
reaches a `generate_image` call once its progress has begun has none, as
[its section](#images-generate_image) says). The agent reads
`model_switching`, `server_not_ready` or any other code as the server says
it. The connector never retries: the agent sees the server's state and
decides.

## The skill

A file cannot go through MCP: a model writes a tool's arguments, so a file's
bytes would pass through the model, and a server reading paths works only on
the machine the harness runs on. So ignis serves a **skill** too, `ignis`: a
`SKILL.md` and Node scripts that run in the harness's own shell, beside the
files, and call ignis's existing routes. ignis touches no file, so it works the
same with ignis local, on the LAN or behind a tunnel. From 0.11.0 the skill
holds three scripts:

| Script | What it does |
|---|---|
| `generate-image.mjs <prompt> <out.png> [--size WxH] [--model id]` | Generates an image through `POST /v1/images/generations` and writes the full-size PNG to `out.png`. |
| `edit-image.mjs <prompt> <out.png> <input...> [--model id]` | Sends the project's pictures (PNG, JPEG or WebP, in order) with the prompt through `POST /v1/images/edits` and writes the PNG to `out.png`. |
| `transcribe.mjs <audio> [--out file]` | Transcribes a recording through `POST /v1/audio/transcriptions` and prints the transcript, or writes it to `--out`. |

The server serves the skill as one `tar.gz` at `GET /v1/connector/skill`, behind
the key and, with `connector.enabled` off, a `404`. It is packed when the
server is built, so it is always the skill that matches the server that gave
it, and its bytes are the same on every request.

Both image scripts pick the image model from the catalog (`GET
/v1/models/catalog`) unless `--model` names one: the model loaded if it is an
image model, else the first on disk that the card can load. With a text model
loaded, naming the image model is what makes the server switch to it, so the
first image can take a while (the switch is the server's, and
`switch.allow_implicit` off refuses it). The scripts wait for the answer: they
use Node's `http` and `https` modules, not `fetch`, which gives up on a
response whose headers take more than about five minutes. They print where the
PNG went, and an error from ignis as the JSON it sent, with a non-zero exit.

The MCP tool `generate_image` covers an image that needs no file: it returns a
preview of at most 512 px and no file. When the image has to be kept, use
`generate-image.mjs`, which writes it at full size.

### Install

One command extracts it into `~/.agents/skills`, where opencode, qwen-code, pi
and DeepSeek Harness read skills. Run it in the shell of the machine the
harness runs on; it reads the two variables below, and with `IGNIS_BASE_URL`
unset it uses a local server. On a server with no key the header is ignored.

On Linux and macOS:

```sh
mkdir -p ~/.agents/skills
curl -fsS -H "Authorization: Bearer $IGNIS_API_KEY" \
  "${IGNIS_BASE_URL:-http://127.0.0.1:8000}/v1/connector/skill" \
  | tar xz -C ~/.agents/skills
```

On Windows, in PowerShell:

```powershell
$base = $env:IGNIS_BASE_URL; if (!$base) { $base = 'http://127.0.0.1:8000' }
$skills = "$HOME/.agents/skills"; $archive = "$env:TEMP/ignis-skill.tgz"
New-Item -ItemType Directory -Force $skills | Out-Null
curl.exe -fsS -H "Authorization: Bearer $env:IGNIS_API_KEY" -o $archive "$base/v1/connector/skill"
tar xzf $archive -C $skills; Remove-Item $archive
```

The PowerShell form uses `curl.exe`, since `curl` is an alias of
`Invoke-WebRequest` there, and downloads to a file before `tar` reads it:
Windows PowerShell 5.1 passes a native command's output through a pipe as
text, which would corrupt the archive. If another `tar` comes first in `PATH`
than Windows's own (Git for Windows's, say, which reads `C:` as a host name),
call `C:\Windows\System32\tar.exe`.

The model gets the same two commands in the connector's `instructions`, a
shorter form of each, so it can install the skill itself when a user hands it
a file.

### The environment

| Variable | Default | What it does |
|---|---|---|
| `IGNIS_BASE_URL` | `http://127.0.0.1:8000` | Where ignis is: the server's LAN address or its tunnel's, as for the MCP entry. |
| `IGNIS_API_KEY` | unset | The server's key, sent as `Authorization: Bearer` only when set. |

A script exits non-zero on a failure. A missing file or an unreachable server
is named in the message, and an error from ignis is printed as the JSON the
server sent.

### After an upgrade

The scripts do not check the server's version: a skill from an older ignis
calls the routes of that version. After upgrading the server, run the install
command again; it replaces the files of the skill, and the folder always
matches the server that gave it.

## The settings

Two fields in the `connector` group ([every field](../reference/configuration.md)).
Both take effect at start only: `PATCH /v1/config` refuses them, so the tool
list depends on how the server started and an agent cannot change it.

| Field | Default | What it does |
|---|---|---|
| `connector.enabled` | `true` | Serves the MCP server at `/v1/mcp` and the skill at `/v1/connector/skill`. Off, both routes answer `404 connector_disabled`. |
| `connector.allow_changes` | `false` | Lists the tools that change the server (its configuration, a model download) beside those that read it. |

`allow_changes` is not a security boundary. It keeps those tools out of the
agent's list, but an agent with a shell and the API key can call
`PATCH /v1/config` or `POST /v1/models/download` itself. What guards the
server is the key ([Access](../configuration/access.md)).
