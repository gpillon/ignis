---
title: "Working in a fork"
description: "Why ignis asks its forks to send fixes and features upstream, and how: an issue, a pull request, a card report."
sidebar:
  order: 7
---

## Why I ask

When I went looking for an inference engine, I found dozens of forks of the
same project, each carrying a feature of its own. To get an engine that did
what I needed, I merged and cherry-picked from a great many of them, and
spent more time carrying other people's patches than writing my own. That is
how ignis began ([Lineage](../../../README.md#lineage)), and I would like it
not to go the same way.

Fork it freely: the license allows it, and a fork is the right place to try
something. But a fix, a feature that is not about your own setup, or a
measurement on a card I do not have is worth far more upstream, where the
next person finds it without hunting through forks.

I read every new issue on [gpillon/ignis](https://github.com/gpillon/ignis/issues).

## What to send upstream

- **A bug in ignis's own code**: an issue with the command you ran (or
  `make config`'s output), what you expected, what happened, and the log
  lines around it. A fix is welcome as a pull request, with its test
  ([Testing](testing.md)).
- **A feature that is not about your setup**: an issue saying what it does and
  why, best before it grows large, so its shape can be agreed while it is
  still cheap to change. Then a pull request, or the fork's commits named in
  the issue.
- **Figures from a card**: the
  [16 GB card reports](https://github.com/gpillon/ignis/issues/360) issue for
  a 16 GB card, an issue of its own for any other.
- **A setting your card or machine needs**: most of that is configuration,
  not code. A [profile of your own](../concepts/profiles.md#defining-your-own),
  a [config file](../configuration/index.md), `local.mk` for make's knobs
  ([Working with make](make.md)), and an
  [operator catalog](../guides/operator-catalog.md) for models of your own
  cover most of it. When configuration cannot say what you need, that is an
  issue too.

A pull request keeps its author on its commits. Work taken from a fork names
the fork and its author in the commit that brings it in.

A fork that stays close to `main` brings its changes back as they are: small
commits, each with its test, and no wholesale reformatting
([`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md)).

## A fork's CI

A fork that keeps GitHub Actions on runs ignis's workflows on its own pushes.
A push to its `main` or to a `ci/**` branch runs the CPU-only tests on Linux
and Windows and both builds, whose archives stay as the run's artifacts; a
push that changes the guide builds it. Nothing is published from a fork: no
container image, no `develop` prerelease, no release on a `v*` tag, no
deployment of the guide. Those run on `gpillon/ignis` alone, so nothing
built in a fork carries the label of an ignis build.

## For an agent working in a fork

A coding agent working in a fork reads `AGENTS.md`, which points here. What
it can do for upstream is notice when a change it makes is not about the
fork: a bug in ignis's own code, or a piece ignis would need anyway. For
each, it can draft the issue for `gpillon/ignis`, as above, and show it to
its user, who posts it or not.

A fork often rewrites `AGENTS.md` for its own workflow. Keeping that file's
pointer to this page keeps the next agent aware of it.
