# Coding standards

The rules a review holds a diff to in this repository. Each one is a
judgement call no tool makes; where an ADR records the decision, the rule
names it and the ADR has the reasons. A rule here overrides the review's
generic smell baseline.

## Diffs are semantic

The tree is not formatter-clean: it has no `rustfmt.toml`, no prettier or
clang-format configuration, and the Makefile deliberately has no `fmt` target.
A diff touches only the lines its change needs and follows each file's
existing wrapping by hand. No repo-wide formatter (`cargo fmt`, prettier,
clang-format) is ever run, before a commit or after a merge; a hunk that only
reformats code the change does not need is a finding.

## Limits are config fields

A margin, limit or threshold that decides what the server does (whether a
load starts, how much it reserves, when it refuses) is a config field with the
measured value as its default, declared once in
`crates/server/src/config/schema.rs` (ADR 0046), never a hardcoded constant.

## Inference is our own Rust

Inference code is the repo's own Rust; C or C++ goes in only where it is truly
necessary, as in the CUDA kernel leaf (ADR 0001). Kernel code is one of ADR
0051's three tiers (ours, an adapted op, a vendored op), and that ADR's last
resort, vendoring a whole third-party project, is refused for a core path; the
speech executor is the precedent (ADR 0068). A library for non-core work, such
as audio file decoding, is fine when it is cheap to build and to carry. A
design that reaches for an external engine lists writing the part in Rust
among its options.

## Extensions get ignis names

OpenAI's names and naming style are for what exists for compatibility with
OpenAI clients. What ignis adds gets an ignis name: `session.ignis.*` for
session fields, `ignis.<event>` for events, `ignis_` for metrics, and it adds
nothing to an OpenAI namespace such as `conversation.item.*` or the `include`
values. A new value inside an OpenAI field is borderline and goes to the
owner. ADR 0069 applies the rule.

## Vendored ops

A vendored op stays the upstream's file byte for byte, or a recorded patch of
it, under its manifest (ADR 0010, `kernel/vendor/VENDOR.md`). Four reasons
admit a recorded patch, made with `scripts/vendor-ninfer.ps1 record-patch`: a
measurement that shows the op is the bottleneck (ADR 0031); a demonstrated
correctness bug, whose patch ships with a test that fails without it and a
`reason` naming the bug and the test (ADR 0037); portability to a CUDA
architecture the build targets, or to a card smaller than the RTX 5090, whose
patch leaves the default `120a` build's device code unchanged and whose
`reason` names the CUDA architecture or the card size and what it fixes: a
compile, a launch that fails at run time there (ADR 0077), or a route choice
the card cannot hold, whose test may run it on the RTX 5090 under a test-only
SM cap (ADR 0091); or a vendored
registry that lacks a case ignis needs, whose `reason` names the shapes
added, why the routes fit them and the test cases that throw without it, and
which a small registry may instead meet by being rewritten as ignis's own
code (ADR 0088). An op rewritten
as ours, or
changed beyond a recorded patch, leaves the vendored tree and
makes no parity claim (ADR 0031, ADR 0051). A comparison with the reference
says whether it crosses an Ignis-patched behaviour.

## A test waits for the whole end state

A test that waits for asynchronous work (a background task, a download, a
switch) polls until every field it then asserts holds, not until the first
sign of the end: fields one reading of the state reports together are often
published at different moments, and a wait on one of them races the others.
The download route tests waited for `on_disk` complete, then asserted
`download` none, and failed about once in a hundred runs (7ac1011f).

## After a merge or a rebase

The checks are `make ci`, whose `check` step compiles the code behind the
`cuda` feature unless it runs with `CUDA=0`; `cargo test` never does. The rest
of what a change needs, a kernel change's GPU legs among it, is
`docs/agents/testing.md`'s "Done means". Resolving a conflict reformats
nothing.

## Skill output and prototypes

- A skill that writes a file into the current directory (a questionnaire, a
  handoff) writes it into the scratchpad, `.scratch/`, which git ignores: such
  a file can carry a partner's internal names, and those reach no commit.
- `/teach` runs in a workspace outside the repo: in the repo it would write its
  own `GLOSSARY.md`, mission file and lessons beside the project's.
- A prototype lives on a throwaway `prototype/<name>` branch, pushed and never
  merged (ADR 0070), exempt from the test and user-guide rules. The issue it
  answers links the branch and records the verdict; the decision it validates
  reaches `main` as ordinary work, with its test and its guide update.

## What the review leaves out

The smell baseline does not apply to:
- **vendored code**: `kernel/vendor/`, `vendor/` and
  `crates/server/assets/swagger-ui/`; the review reads the manifest entry or
  the recorded patch instead;
- **ABI structs**: the `#[repr(C)]` mirrors of `kernel/include/*.h`, such as
  `crates/core/src/step_abi.rs` and `crates/artifact/src/ffi.rs`, whose fields
  the C ABI fixes (ADR 0001, ADR 0009, ADR 0016); a data clump there is the
  ABI;
- **generated reference docs**: the pages under `docs/user/reference/` marked
  generated, which `make docs-reference` writes from the code; the review
  reads the code that produces them.

Tooling already enforces the kernel's vendored tree against its manifest and
the generated pages against the code (both in `cargo test`), the guide's
links (`make docs-build`), and the commit messages' conventions
(`.githooks/commit-msg`). The guardrails are `make ci`, run by hand before a
merge, and the release workflow's `cargo test` on Linux and Windows on every
code push to `main`.
