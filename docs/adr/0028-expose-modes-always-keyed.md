# ADR 0028 — exposure is a named mode, and an exposed server always has a key

## Status

Accepted (2026-09-15, owner decision).

## Decision

`ignis-server --expose <mode>` (`IGNIS_EXPOSE`) makes the server reachable
from outside this machine without opening a port. The first mode is
`cloudflare-quick`: an anonymous Cloudflare quick tunnel
(`https://<random>.trycloudflare.com`), spoken natively by the
`cloudflare-quick-tunnel` crate (QUIC + capnp-RPC, no `cloudflared` child
process).

- **A mode, not a switch.** The flag takes a value so a later way of
  exposing the server (a named tunnel, a reverse proxy) is a new variant of
  `ignis_server::expose::Expose`, not a new flag.
- **Exposed means keyed.** With `--expose` set and no `--api-key`, the server
  behaves exactly as `--api-key auto`: it generates a key and prints it. A
  key the operator named is kept. There is no way to expose an open API.
- **Opened after bind, before serving.** The tunnel forwards to
  `127.0.0.1:<bound port>`, so `--bind` must accept IPv4 loopback
  (`127.0.0.1` or `0.0.0.0`). If the tunnel cannot open, the start is
  refused — a server the operator believes is reachable but is not is worse
  than no server.
- **The URL is printed on stdout** (`ignis-server: public URL: …`, plus the
  Playground's `/ui/` when `--ui` is on), next to the generated key; the
  Makefile helpers repeat both on the console. `make … EXPOSE=cloudflare-quick`.
- The tunnel is closed after the HTTP drain on shutdown, so in-flight remote
  requests finish through it.

## Considered Options

- **Spawn `cloudflared`** — rejected: a ~30 MB external binary to install or
  download, and a URL scraped from its stderr.
- **`--expose` with an open API allowed** — rejected: a quick-tunnel URL is
  public, and the engine behind it is a single GPU.
