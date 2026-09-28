"""EXPERIMENT (branch locate-long-context): reversible compression of a log
for `locate` — the owner's idea (2026-09-27): fold near-duplicate lines,
search the folded text, unfold to the original line.

A log is clustered into **templates**, Drain-style: lines of the same source
label and the same token count whose tokens agree on at least `SIM` of the
positions share a cluster, and every position where they differ becomes
`<*>`. Obvious variables (times, numbers, hex, ids, addresses) are masked
before the comparison, so they never split a cluster.

- **level 1**: one line per cluster — its template and `(xN)` — the text a
  first `locate` searches for the kind of line;
- **level 2**: the chosen cluster's lines rendered as **their variable
  values only** (the tokens at the template's `<*>` positions, and the
  line's time), exact repeats folded into one line with `(xN)` — the text a
  second `locate` searches for the instance;
- the map back: (cluster, level-2 row) -> the original line indices.

Every function is pure; `Folded` keeps what unfolding needs.
"""

import re
from dataclasses import dataclass, field

SIM = 0.5
TOKEN = re.compile(r"\S+")
TIME = re.compile(r"\d{4}[-/]\d\d[-/]\d\d[T ]\d\d:\d\d(:\d\d(\.\d+)?)?(Z|[+-]\d\d:?\d\d)?|\b\d\d:\d\d:\d\d(\.\d+)?\b")
VARIABLE = re.compile(
    r"^(\d+([.,:/]\d+)*(ms|s|µs|us|m|h|%|kb|mb|b)?|[0-9a-f]{8,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|\d+\.\d+\.\d+\.\d+(:\d+)?)$", re.I)
LABEL = re.compile(r"^\[[^\]]*\]")


def split_label(line):
    """A bracket-opened prefix is the line's source label (`[svc-a]`) —
    unless it holds a time: then it is the line's timestamp
    (`[Sun Dec 04 04:47:44 2005]`, `[10.30 16:49:06]`), and a label read
    from it would put every line in a template of its own (spec 22,
    GitHub #278, found on R3's first run)."""
    m = LABEL.match(line)
    if m and not TIME.search(m.group(0)):
        return m.group(0), line[m.end():]
    return "", line


def stamp_of(line):
    m = TIME.search(line)
    return m.group(0) if m else ""


def tokens_of(body):
    """Tokens with the line's time removed and obvious variables masked:
    what clustering compares (`<v>` never splits a cluster)."""
    body = TIME.sub(" ", body)
    toks = TOKEN.findall(body)
    return [("<v>" if VARIABLE.match(t.strip('",;()[]{}=')) else t) for t in toks], TOKEN.findall(TIME.sub(" ", body))


@dataclass
class Cluster:
    label: str
    template: list
    members: list = field(default_factory=list)   # original line indices
    raw: list = field(default_factory=list)       # each member's raw tokens (for its values)


@dataclass
class Folded:
    lines: list
    clusters: list
    level1: list          # level-1 text per cluster, in first-seen order
    order: list           # cluster index per level-1 row


def summarize(c, cap=6, width=24, budget=600):
    """A cluster's template with each variable slot showing its distinct
    values — at most `cap`, each cut to `width` characters, `+N` for the
    rest — so a word the question names stays visible in its cluster."""
    out = []
    for k, t in enumerate(c.template):
        if t not in ("<*>", "<v>"):
            out.append(t)
            continue
        seen = []
        for raw in c.raw:
            v = raw[k][:width] if k < len(raw) else ""
            if v not in seen:
                seen.append(v)
        # every distinct value while they fit the budget, else the first `cap`
        keep = seen if sum(len(v) + 1 for v in seen) <= budget else seen[:cap]
        shown = "|".join(keep) + (f"|+{len(seen) - len(keep)}" if len(seen) > len(keep) else "")
        out.append("{" + shown + "}" if len(seen) > 1 else (seen[0] if seen else "<*>"))
    return out


def fold(lines, sim=SIM, values=False):
    by_key = {}
    clusters = []
    for i, line in enumerate(lines):
        label, body = split_label(line)
        masked, raw = tokens_of(body)
        key = (label, len(masked))
        best, best_score = None, -1.0
        for c in by_key.get(key, []):
            same = sum(a == b for a, b in zip(c.template, masked) if a != "<*>")
            score = same / max(1, len(masked))
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= sim:
            best.template = [a if a == b else "<*>" for a, b in zip(best.template, masked)]
            best.members.append(i)
            best.raw.append(raw)
        else:
            c = Cluster(label=label, template=list(masked), members=[i], raw=[raw])
            by_key.setdefault(key, []).append(c)
            clusters.append(c)
    clusters.sort(key=lambda c: c.members[0])
    level1 = []
    for c in clusters:
        shown = summarize(c) if values else c.template
        text = " ".join(t if t != "<v>" else "<*>" for t in shown)
        count = f" (x{len(c.members)})" if len(c.members) > 1 else ""
        level1.append(f"{c.label} {text}{count}".strip())
    return Folded(lines=lines, clusters=clusters, level1=level1, order=list(range(len(clusters))))


def _common_affixes(values):
    """The prefix and suffix every value of a slot shares (`application=`,
    a closing quote), so a row shows only what differs."""
    if len(values) < 2:
        return 0, 0
    pre = 0
    while all(len(v) > pre for v in values) and len({v[pre] for v in values}) == 1:
        pre += 1
    suf = 0
    while all(len(v) - pre > suf for v in values) and len({v[-1 - suf] for v in values}) == 1:
        suf += 1
    return pre, suf


def level2(folded, ci, values_first=True):
    """The chosen cluster's lines as their variable values, exact repeats
    folded: (rows, the original line indices of each row). With
    `values_first` the slots' shared affixes are cut and the time goes last,
    so a row begins with what tells it apart."""
    c = folded.clusters[ci]
    slots = [k for k, t in enumerate(c.template) if t in ("<*>", "<v>")]
    cut = {}
    if values_first:
        for k in slots:
            cut[k] = _common_affixes([raw[k] for raw in c.raw if k < len(raw)])
    rows, index = [], {}
    for member, raw in zip(c.members, c.raw):
        parts = []
        for k in slots:
            if k >= len(raw):
                continue
            v = raw[k]
            pre, suf = cut.get(k, (0, 0))
            v = v[pre:len(v) - suf] if len(v) > pre + suf else v
            if v:
                parts.append(v)
        values = " | ".join(parts)
        stamp = stamp_of(folded.lines[member])
        key = values
        if key in index:
            rows[index[key]][1].append(member)
            continue
        index[key] = len(rows)
        if values_first:
            text = f"{values} @ {stamp}" if stamp else values
        else:
            text = f"{stamp} | {values}" if stamp else values
        rows.append([text, [member]])
    texts = [text + (f" (x{len(m)})" if len(m) > 1 else "") for text, m in rows]
    return texts, [m for _, m in rows]


def stats(lines, sim=SIM):
    f = fold(lines, sim)
    sizes = sorted((len(c.members) for c in f.clusters), reverse=True)
    return {"lines": len(lines), "clusters": len(f.clusters), "singletons": sum(s == 1 for s in sizes),
            "largest": sizes[:5]}
