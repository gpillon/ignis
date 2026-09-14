# make — the development entry point

The root `Makefile` builds, runs and checks ignis on every OS through one
platform-neutral target graph; per-OS behaviour sits behind the hooks in
`mk/os/<os>.mk`. The decision and its deliberate behaviours are ADR 0027.

Windows implements every hook. Linux is scaffolded: the CPU mock, web and
test targets go through the neutral graph, and each hook it cannot run yet
fails naming itself.

## Specs

- `specs/01-linux-hooks.md` — implement the Linux hooks, so `CUDA=1` builds,
  the GPU checks and the server session targets work on a Linux host.
