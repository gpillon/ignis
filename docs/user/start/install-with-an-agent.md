---
title: "Install with a coding agent"
description: "Let the coding agent you already use install ignis: what to tell it, what it checks, what it asks you, and what it leaves to you."
sidebar:
  order: 5
---

A coding agent that can run commands and read web pages (opencode, Claude
Code, Codex and the like) can install ignis for you. The repository's README
points an agent, on its first screen, to a page written for it:
[the agent path](../paths/agent.md), with the checks to run, the choices that
are yours and the commands of each step.

## What to tell it

```
Install ignis from https://github.com/gpillon/ignis
```

The same page answers the other things you might ask it:

- "What is https://github.com/gpillon/ignis?"
- "Will ignis run on this machine?"
- "Point opencode at the ignis server on this machine."

## Test a 16 GB card

The `rtx-16gb` profile has never run on a real 16 GB card. With an RTX 5060
Ti 16 GB, a 5070 Ti or a 5080, your agent can install ignis, test it and
write a report for [issue #360](https://github.com/gpillon/ignis/issues/360):

```
I have a 16 GB RTX 50 card. Read https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/paths/agent.md and follow its section on testing a 16 GB card: help me install ignis, test it, and write the report. Ask me before downloading or installing anything.
```

It shows you its plan first, asks before it downloads the 13.4 GB model, tries
the default and then a few settings if the default is refused, and writes the
report in English for you to read before anything is posted. An advanced part,
about 30 minutes, measures the context ceiling, one lane against two and
DFlash2 on against off, if you want it. At the end it asks whether to keep
ignis or remove what it installed
([the agent path's test](../paths/agent.md#testing-a-16-gb-card)).

## What it does

1. **Checks the machine**: the card, the driver, the system, the disk and,
   for Flash-Next, the host RAM. On a card ignis does not support it stops
   and tells you why ([What it runs on](index.md#what-it-runs-on)).
2. **Asks you** which model, where to keep it (tens of GiB), the release
   archive or the container, and whether the server should be reachable from
   other machines.
3. **Installs**: downloads the latest release, checks its SHA-256, fetches the
   model you chose into the directory you chose, starts the server and waits
   until `/ready` answers.
4. **Connects a client** when you ask, itself included: ignis is an
   OpenAI-compatible endpoint ([Connect a client](connect-a-client.md)).

## What it leaves to you

The agent path tells it to ask before any of these:

- downloading a model;
- installing a driver or the CUDA runtime, or anything that needs
  administrator rights;
- stopping a process that holds the card;
- making the server reachable beyond this machine;
- building from source instead of taking a release;
- posting a report, or anything else, to GitHub;
- writing a config file, or deleting what it installed.

It matters for the download in particular: started from an agent's shell,
with no terminal to answer, the server fetches its model without asking
([Letting the server fetch it](../guides/models.md#letting-the-server-fetch-it)).

:::caution
The agent runs commands on your machine with your rights. Read what it
proposes before you approve it, as for any install.
:::
