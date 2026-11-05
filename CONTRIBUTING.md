# Contributing to ignis

Start with the guide's contributing section, published at
**https://gpillon.github.io/ignis/contributing/** and readable here under
[`docs/user/contributing/`](docs/user/contributing/index.md): the ground
rules, the repository's layout, building from source, the tests and the GPU
discipline, make, and how work is tracked.

In short:

- Working in a fork: send upstream what is not about your setup, a fix, a
  feature or a card's figures ([Working in a fork](docs/user/contributing/forks.md)).
- Report a bug or ask for a feature in a [GitHub issue](https://github.com/gpillon/ignis/issues):
  the command you ran (or `make config`'s output), what you expected, what
  happened, and the server's log lines around it.
- Every code change ships with a test; the work is done when `cargo test`
  passes workspace-wide.
- Every user-visible change updates the user guide, `docs/user/`, in the same
  commit ([`docs/agents/user-docs.md`](docs/agents/user-docs.md)).
- One run at a time on the GPU: `make gpu-status` before any of them.
- The rules a review holds a diff to are in
  [`CODING_STANDARDS.md`](CODING_STANDARDS.md).
- The conventions agents follow — and humans too — are in
  [`AGENTS.md`](AGENTS.md) and [`docs/agents/`](docs/agents/).
