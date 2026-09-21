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

The version lives where every other frontend dependency's does —
`web/package.json` (`swagger-ui-dist`, pinned exactly). The files here are a
copy of what that pin installs:

```sh
npm --prefix web install --save-dev --save-exact swagger-ui-dist@<version>
make swagger-ui-sync     # copies the files here and rewrites the line above
make ci                  # swagger-ui-check fails if the two ever disagree
```

`make swagger-ui-check` runs inside `make ci`, comparing every file here
against `web/node_modules/swagger-ui-dist`, so a bump that was not synced —
or a hand-edited file here — fails the gate instead of shipping. The page
that drives these files is `crates/server/src/openapi.rs`'s `PAGE_HTML`, so
a new Swagger UI major is worth a look at
`http://127.0.0.1:8000/v1` before committing.
