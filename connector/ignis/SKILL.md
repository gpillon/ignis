---
name: ignis
description: Use the ignis inference server from the shell, beside the user's files, to transcribe a recording to text, generate an image into a file, or edit the project's pictures. Use it when the user gives an audio file, asks what was said in one, wants an image saved in the project, or wants a picture of the project changed.
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

## generate-image

`node scripts/generate-image.mjs <prompt> <out.png> [--size WxH] [--model id]`

Generates an image from the prompt and writes the full-size PNG to `out.png`.
Use it when the image has to be a file in the project: a logo, an asset, a
mock-up. `--size` is one of the sizes the server serves (`1024x1024`,
`2048x2048` and the like, or `auto`); without it the server picks. Without
`--model` the script picks the image model from the server's catalog: the one
loaded, else the first on disk. The server may switch models to make the
image, which takes a while; the script waits for it.

If the image needs no file, because it is only to look at, call the ignis
server's tool `generate_image` instead: it returns a small preview and needs
no script.

## edit-image

`node scripts/edit-image.mjs <prompt> <out.png> <input...> [--model id]`

Edits the project's own pictures: the prompt says the change, each `input` is
a PNG, JPEG or WebP file (several may be given, in order), and the result is
written as a PNG to `out.png`, never over an input unless you name it. Use it
to restyle, recolour or amend a picture that is already in the project. The
model is picked as for `generate-image`.
