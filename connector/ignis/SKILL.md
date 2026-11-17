---
name: ignis
description: Use the ignis inference server from the shell, beside the user's files, to transcribe a recording to text. Use it when the user gives an audio file, or asks what was said in one.
---

# ignis

ignis is the inference server this agent's model may run on. Its scripts run
in your shell and call its HTTP routes, so a file on this machine reaches
ignis whether the server is local, on the LAN or behind a tunnel, and ignis
never touches a file itself.

Every script needs Node 18 or later and nothing else. They read two
environment variables:

- `IGNIS_BASE_URL`: where ignis is, `http://127.0.0.1:8000` when unset.
- `IGNIS_API_KEY`: the server's key, sent as `Authorization: Bearer` only when
  set. A server started with a key refuses a call without it.

A script exits non-zero on a failure. A missing file or an unreachable server
is named in the message; an error from ignis is printed as the server's JSON.

## transcribe

`node scripts/transcribe.mjs <audio> [--out file]`

Transcribes a recording (wav, mp3, flac, ogg, m4a, webm and the like) and
prints the transcript. With `--out file` it writes the transcript there
instead and prints nothing. Use it when the user hands over a recording or
asks what was said in one. It runs on the server's CPU beside the text model,
so it works while a model is switching.
