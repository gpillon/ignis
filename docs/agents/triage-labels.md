# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

A state label also sets the issue's Status on the project board: **Project board** in `docs/agents/issue-tracker.md`.

## Category roles

| Role          | Label in our tracker |
| ------------- | -------------------- |
| `bug`         | `bug`                |
| `enhancement` | `enhancement`        |

A new spec issue and each of its tickets carry one category beside
`ready-for-agent`, `enhancement` for new work or `bug` for a defect, since
`/to-spec` and `/to-tickets` apply only the state role.

## Repo-only labels

Two labels this repo adds to the five roles above. Each is added beside the
issue's other labels, and its state role and category stay as they are.

| Label   | Kind     | Meaning |
| ------- | -------- | ------- |
| `idea`  | marker   | Parked: kept open, not specified, not scheduled. Add it to a proposal nobody will build soon, and keep the state role it has (often `ready-for-human`). When picked up, remove `idea` and take the issue into `/grill-with-docs`; close it with a link to the spec or tickets it became. |
| `quick` | marker   | The owner's decision is fast and the work after it is small. It combines with any state and with `idea`. When showing what needs attention, list open `quick` issues as a fourth bucket. |

## Other labels

`Bucket 1`, `Bucket 2`, `dependencies`, `javascript`, `python`, `rust`,
`documentation`, `accessibility`, `good first issue`, `help wanted`, `question`,
`duplicate` and `invalid` are not roles: triage leaves them as they are.

