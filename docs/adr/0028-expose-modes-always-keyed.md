---
status: accepted
---

# Exposure is a named mode, and an exposed server always has a key

`--server-expose <mode>` makes the server reachable from outside this machine without opening a port. The first mode,
`cloudflare-quick`, is an anonymous quick tunnel with a random public URL, spoken natively by a crate, with no child
process.

- **A mode, not a switch.** The flag takes a value so that a later way of exposing the server (a named tunnel, a
  reverse proxy) is a new variant of the mode, not a new flag.
- **Exposed means keyed.** With a mode set and no API key, the server behaves exactly as with the key `auto`: it
  generates a key and prints it. A key the operator named is kept. There is no way to expose an open API: a
  quick-tunnel URL is public, and the engine behind it is a single GPU.
- **Opened after bind, before serving.** The tunnel forwards to `127.0.0.1:<bound port>`, so the API's bind address must
  accept IPv4 loopback (`127.0.0.1` or `0.0.0.0`). If the tunnel cannot open, the start is refused: a server the
  operator believes is reachable but is not is worse than no server.
- **The URL is printed on stdout**, next to the generated key, with the Playground's address when the Playground is
  served; the Makefile helpers repeat both on the console.
- **Closed after the HTTP drain** on shutdown, so in-flight remote requests finish through the tunnel.

## Considered Options

- **Spawning the tunnel vendor's own client.** Rejected: an external binary of about 30 MB to install or download,
  and a URL scraped from its stderr.
