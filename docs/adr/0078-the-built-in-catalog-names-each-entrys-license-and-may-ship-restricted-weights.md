---
status: accepted
---

# The built-in catalog names each entry's license and may ship weights whose license restricts commercial use

Swift 1.5 (GitHub #449) is a fine-tune of the 27B under the Swift Open License
v1.0, which leaves commercial use by an organization above US$1M of revenue
unlicensed. Until then the built-in catalog had held weights under open licenses
and Qwen's own (Flash-Next's community license, the image model's research
license), with no entry saying which. The built-in catalog takes it, and every
catalog entry now states its license (`license`: the license's identifier and a
link to its text), required in the built-in catalog and optional in an
operator's, and shown beside the entry in the Server view, `model list` and
`model download`, with no acceptance step: whoever downloads is bound by the
license, as with any download from the hub. Publishing such weights only for an
operator catalog to name was rejected, since users would then hand-write pins
for an artifact the project built and gated itself; a click-through acceptance
was rejected as a gate ignis cannot enforce and the hub does not ask for.
