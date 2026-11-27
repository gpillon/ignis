---
status: accepted
---

# The server fetches its own model, and asks first only when somebody is there

A server started with no artifact came up on the placeholder template and the CPU mock: a server that answers, in
text that is not natural language, and says nothing about where the real model comes from, while the models are
published and the one thing between a fresh machine and a working server was a download line in a README. So
`ignis-server` fetches a missing model itself at start, from the catalog (ADR 0047). With stdin on a terminal it
asks first, and anything but a yes is a no, EOF included, because a model's tens of gigabytes are not something to
spend on a stray Enter. With no terminal, a container, a daemon or CI, there is nobody to ask and it just
downloads: a start that blocks forever on a question nobody will answer is worse than either answer.

- **Only when no artifact is named.** A path the operator named is their word: missing, it refuses the start as it
  always has. A typo must stay a typo rather than become a download fetched somewhere else.
- **Only on a build with `cuda`.** A binary that cannot run the weights has no use for them. This, not a
  remembered flag, is what keeps the mock, the tests and every CPU CI job off the network.
- **A flat destination**: the repo's file names under the download path, exactly what a download from the hub by
  hand produces, so a model fetched by hand and one fetched by the server are the same file, and a machine that
  already has it downloads nothing.
- **The pins are the trust anchor.** A built-in entry's digest is in the binary, never read from the repo: the
  digest the repo publishes sits in the same trust domain as the artifact it describes, so it can attest nothing
  about it. A file is in place only once its length and its digest match its pins, so bytes republished under the
  same file name fail, by length or by digest. That is the intended direction of the failure: a new artifact is a
  new catalog entry and a new binary, not a silent swap.
- **A failed fetch refuses the start.** Falling back to the mock after the operator said yes would be the same
  silent degradation the loader refuses for an unclean checksum.

## Considered Options

- **A model list next to the binary as the trust anchor.** A file an attacker can edit is not a trust anchor, and a
  catalog that ships with the engine cannot drift from the loader that consumes it. An operator's own file adds
  entries under the operator's word and never re-pins a built-in id (ADR 0047).
- **Trusting the digest the repo publishes.** A digest served by the same repo, over the same connection, as the
  artifact it describes attests only that the transfer was not corrupted, which the length check already tells,
  and nothing about what was published.
- **Asking even without a terminal**, with a timeout that defaults to no. A container start that waits, then comes
  up wrong, is the worst of both.
