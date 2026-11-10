# ignis patch on cloudflare-quick-tunnel 0.3.1

Upstream: https://github.com/lordmacu/cloudflare-quick-tunnel-rs (crates.io
`cloudflare-quick-tunnel` 0.3.1, MIT OR Apache-2.0). Wired in through
`[patch.crates-io]` in the workspace `Cargo.toml`. Two changes: the
chunked-response fix in `src/proxy.rs`, and a dependency bump (`Cargo.toml`,
`src/edge.rs` and the regenerated `src/proto_gen/`).

## The chunked-response fix

### The bug

An origin response with `Transfer-Encoding: chunked` — what axum/hyper send
for every body of unknown length, so every SSE stream from
`/v1/chat/completions` — was forwarded to the edge as raw bytes until the
local socket reached EOF:

- the chunk-size lines (`B6\r\n`, the final `0\r\n\r\n`) arrived at the
  client inside the body;
- the socket was opened `Connection: keep-alive`, so the origin never closed
  it after the last chunk, and the response hung until an idle timeout.

### The fix

When the response is chunked (and not an upgrade), the proxy drops the
`Transfer-Encoding` header — the edge frames the body itself — decodes the
chunks incrementally, forwards each chunk's data as soon as it arrives, and
closes the edge stream at the terminal chunk instead of waiting for EOF.
`ChunkedDecoder` has unit tests for split input, extensions, trailers and
bad sizes.

## The dependency bump (GitHub #568)

Upstream 0.3.1 asks for `capnp`, `capnp-futures` and `capnp-rpc` 0.20 and for
`hickory-resolver` 0.24, and Dependabot does not cross a requirement, so it
could not fix the alerts they carried (the repository's Dependabot alerts
#1 to #6 and #58 to #60): `capnp` 0.20.6 (#1, critical, fixed in 0.24.0),
`hickory-proto` and `hickory-resolver` 0.24.4 (#5 and #6, fixed in 0.26.1 and
0.26.2), and `rustls` 0.21.12 (#58 to #60) and `rustls-webpki` 0.101.7 (#2 to
#4), which only hickory 0.24's `tokio-rustls` 0.24 brought in. The vendored
`Cargo.toml`, the manifest cargo reads, now asks for:

- `capnp`, `capnp-futures` and `capnp-rpc` 0.27 (locked at 0.27.3, 0.27.0
  and 0.27.5);
- `hickory-resolver` 0.26.2 (locked at 0.26.3), still with
  `default-features = false`, and the features `tokio` and `tls-ring`, which
  replace 0.24's `tokio-runtime` and `dns-over-rustls`: `system-config` stays
  off and ring stays the only crypto provider (the `*-aws-lc-rs` features
  would add a C build);
- `webpki-roots` 1, for the DoT roots below;
- `rust-version` 1.88, hickory 0.26's.

`Cargo.toml.orig` is left as upstream published it.

### Edge discovery (`src/edge.rs`)

Ported to hickory 0.26's `TokioResolver` and its builder, with 0.24's
behaviour:

- the primary resolver is hickory's Google public group over UDP and TCP on
  port 53 (`ResolverConfig::udp_and_tcp(&GOOGLE)`), which is what 0.24's
  `ResolverConfig::default()` was: not the host's configuration;
- the fallback is DNS over TLS to 1.1.1.1:853, named `cloudflare-dns.com`,
  with a 15 s timeout, tried when the primary fails (a primary that fails to
  build counts) or returns no edges;
- both ask for A records first and for AAAA only when that fails
  (`Ipv4thenIpv6`, 0.24's default): 0.26's default asks for both, which
  would add IPv6 edges the list never held. The other options follow 0.26's
  defaults (EDNS0 on, a cache of 8,192 entries rather than 32): they change
  how a query travels, not which edges come back.

`discover`'s doc comment, which said "System resolver first", and its two
`warn!` lines now say "primary resolver": the primary never read the host's
configuration.

The DoT roots: in 0.24.4's source the DoT client's root store is filled only
by hickory's `webpki-roots` or `native-certs` features, and the crate enabled
neither, so the fallback started from an empty store. With an empty store
the handshake with 1.1.1.1 fails: `live_dot_fallback_returns_edges` fails
when the roots are taken out. The fallback is now given a rustls
`ClientConfig` with the ring provider and the `webpki-roots` crate's roots,
the bundle hickory's own `webpki-roots` feature would load, through
`with_tls_config`. The crate builds that store itself, rather than enabling
hickory's feature, so that a unit test can see it is not empty.

Tests: `primary_is_googles_public_group_over_udp_and_tcp`,
`dot_fallback_is_cloudflare_over_tls_on_853`, `both_resolvers_ask_ipv4_first`,
`dot_fallback_trusts_the_public_roots`, and the opt-in live
`live_dot_fallback_returns_edges` beside `live_discover_returns_edges`.

### The Cap'n Proto bindings (`src/proto_gen/`)

Generated code is bound to its runtime's version, so both files were
regenerated from `schemas/` with `capnpc` 0.27.0 and the `capnp` compiler
0.9.2 (Debian bookworm's `capnproto` 0.9.2-2); each file's header records
both. The recipe is upstream's `scripts/regen-schemas.sh`, which the crate
package does not ship. In a `docker.io/library/rust:1-bookworm` container,
with `schemas/` mounted at `/work/schemas`:

```sh
apt-get update && apt-get install -y capnproto
cargo install capnpc --version 0.27.0 --locked
mkdir -p /work/out
capnp compile -orust:/work/out --src-prefix=/work/schemas \
    /work/schemas/tunnelrpc.capnp /work/schemas/quic_metadata_protocol.capnp
```

then copy `/work/out/{tunnelrpc,quic_metadata_protocol}_capnp.rs` over
`src/proto_gen/`. The modules stay at the crate root (`src/lib.rs`), since
the generator emits absolute `crate::<schema>_capnp::` paths. The
hand-written capnp users (`rpc.rs`, `stream.rs`) needed no change from 0.20
to 0.27.

### Running the crate's tests

`vendor/` is excluded from the workspace, so neither `cargo test` nor CI runs
them. From the repository root, against the crate's own manifest and with the
workspace's lockfile beside it, so the run tests the versions that ship:

```sh
cp Cargo.lock vendor/cloudflare-quick-tunnel/
cargo test --manifest-path vendor/cloudflare-quick-tunnel/Cargo.toml --target-dir <outside the repo>
# the live tests (edge discovery, the DoT fallback, a QUIC handshake):
CFQT_LIVE_TESTS=1 cargo test --manifest-path vendor/cloudflare-quick-tunnel/Cargo.toml --target-dir <same> -- --ignored
rm vendor/cloudflare-quick-tunnel/Cargo.lock
```

No `Cargo.lock` or `target/` under `vendor/` is committed.

## Dropping the patch

Drop the patch once upstream decodes chunked responses and asks for `capnp`
0.24.0 or later and `hickory-resolver` 0.26.2 or later. Until then, the next
bump of these dependencies repeats the steps above.
