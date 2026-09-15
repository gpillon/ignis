# ignis patch on cloudflare-quick-tunnel 0.3.1

Upstream: https://github.com/lordmacu/cloudflare-quick-tunnel-rs (crates.io
`cloudflare-quick-tunnel` 0.3.1, MIT OR Apache-2.0). Wired in through
`[patch.crates-io]` in the workspace `Cargo.toml`; the only change is in
`src/proxy.rs`.

## The bug

An origin response with `Transfer-Encoding: chunked` — what axum/hyper send
for every body of unknown length, so every SSE stream from
`/v1/chat/completions` — was forwarded to the edge as raw bytes until the
local socket reached EOF:

- the chunk-size lines (`B6\r\n`, the final `0\r\n\r\n`) arrived at the
  client inside the body;
- the socket was opened `Connection: keep-alive`, so the origin never closed
  it after the last chunk, and the response hung until an idle timeout.

## The fix

When the response is chunked (and not an upgrade), the proxy drops the
`Transfer-Encoding` header — the edge frames the body itself — decodes the
chunks incrementally, forwards each chunk's data as soon as it arrives, and
closes the edge stream at the terminal chunk instead of waiting for EOF.
`ChunkedDecoder` has unit tests for split input, extensions, trailers and
bad sizes.

Drop the patch once upstream decodes chunked responses.
