# Vendored Swagger UI

`swagger-ui-dist` **5.33.0**, from npm, unmodified. Apache-2.0 (`LICENSE`,
`NOTICE`, `swagger-ui-bundle.js.LICENSE.txt` — the bundle's own
third-party notices, served beside it).

Two files of the package are kept: `swagger-ui-bundle.js` and
`swagger-ui.css`. The page that drives them is
`crates/server/src/openapi.rs`'s own `PAGE_HTML`, not the package's
`index.html`, because the document's URL and the `Authorize` button's
behaviour are ours to state. The source maps, the ES bundles, the
standalone preset (the topbar this page does not show) and the OAuth
redirect are not kept.

## Why vendored here rather than through `utoipa-swagger-ui`

That crate embeds the same distribution through `rust-embed`, whose proc
macro pulls `walkdir → winapi-util → windows-sys 0.59` — and `windows-sys`
0.59 builds its raw-dylib imports with `dlltool`, which the
`x86_64-pc-windows-gnu` *host* toolchain on this project's dev machine
cannot run (the `rust-mingw` `dlltool.exe` fails without the rest of
binutils). The dependency is a **host** one, so the MSVC target pin in
`.cargo/config.toml` does not avoid it: `cargo check -p ignis-server`
fails outright. Vendoring the two files costs a binary no bigger than the
crate's own embed, and builds with any toolchain, offline, on every
platform.

## Updating

```sh
npm pack swagger-ui-dist@<version>
tar xzf swagger-ui-dist-<version>.tgz
cp package/swagger-ui.css package/swagger-ui-bundle.js \
   package/swagger-ui-bundle.js.LICENSE.txt package/LICENSE package/NOTICE \
   crates/server/assets/swagger-ui/
```

Then bump the version at the top of this file and run
`cargo test -p ignis-server --test openapi_http`, which loads the page and
its assets through the router.
