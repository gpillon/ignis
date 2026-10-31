"""GitHub #429 (spec #428): three ways of keeping a transcript's notes, replayed
on the live server at the Transcribe view's cadence, each told the transcript's
context as #440 tells it.

    R  the running notes: one append-only thread with the text model, four
       tools (add_line, replace_line, remove_line, rewrite_section only on a
       full section), one-line results, the kind's live sections each with a
       cap, the page's guards, a cap of rounds per update, times in total
       minutes, a restart from a snapshot past a threshold; the look runs
       between updates, as the view runs it;
    A  today's full rewrite at the same triggers, the look between them (a pass
       of its own, so A and R never share the server's prompt cache);
    C  the recap: one writing over the whole transcript at the end.

    python scripts/notes-replay/replay.py <transcript .md export> <work|general|game|lesson> <label> [--me "Speaker 1"] [--ways R,A,C] [--restart-tokens 64000]

The transcript is a Transcribe `.md` export (feed.py gets one from a recording
without a download). The prompts are `prompts.json`, the view's as of
2026-10-31, frozen; the running notes' and the recap's are below. Outputs go to
`.scratch/notes-replay/<label>/`, summed up by summary.py. Standard library
only. Run it only on public recordings.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b"
P = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
SETTINGS = {k: v for k, v in P["notes"]["rest"].items() if k not in ("stream", "stream_options")}
ANALYSIS_SETTINGS = {k: v for k, v in P["analysis"]["rest"].items() if k not in ("stream", "stream_options")}
ASSIST = P["assistSystem"]
# #440 names the language to write in, in the context block; the frozen tasks still carry the old rule.
LANG_OLD = "in the same language as the transcript: a transcript in Italian gets notes in Italian, one in German notes in German."
LANG_NEW = "in the language the context says to write in."

MAX_ROUNDS = 6          # rounds of one update; what is left goes to the next update
LINE_CHARS = 240        # a line longer than this is refused

# The running sections per kind (Q186): the current state, or facts that add up. Name, cap, ask.
RUNNING = {
    "work": [
        ("Now", 1, "what is being discussed right now, in one line"),
        ("People", 20, "one line a person who spoke or was named: who they are, their role or stance as the conversation shows it. Add a person the first time they come up, complete their line as more is learned, never remove one"),
        ("Figures & dates", 12, "every figure said (an amount, a percentage, a quantity, a date, a deadline), one a line, with what it refers to"),
        ("Asked of me", 6, "questions or requests put to Me and not answered yet; remove a line once Me has answered it"),
        ("Commitments heard", 10, "who said they would do what, and by when if it was said"),
    ],
    "general": [
        ("Now", 1, "what is being talked about right now, in one line"),
        ("People", 20, "one line a person who spoke or was named: who they are, as the conversation shows it. Add a person the first time they come up, never remove one"),
        ("Points", 10, "the points made, one a line, with who made them"),
        ("Asked of me", 6, "questions or requests put to Me and not answered yet; remove a line once Me has answered it"),
        ("Open questions", 8, "what was raised and is still open; remove a line once it is settled"),
    ],
    "game": [
        ("Now playing", 1, "the game being played and where it stands, in one line"),
        ("Players & sides", 12, "one line a player: their side, team or role"),
        ("Score", 8, "the score or the standings as they are now"),
        ("Moments", 10, "the moments that mattered, with their time"),
        ("Rules in question", 6, "rules asked about or disputed, and how they were settled"),
    ],
    "lesson": [
        ("Now", 1, "the subject being taught right now, in one line"),
        ("Concepts", 12, "the concepts taught, one a line, with a line of explanation"),
        ("Terms", 20, "the terms introduced, one a line, with what they mean"),
        ("Questions", 8, "the questions asked, with the answer if one was given"),
        ("Homework & dates", 8, "homework, readings, exams and their dates"),
    ],
}

# The recap's additions to today's sections (Q187), each after the named section.
RECAP_EXTRA = {
    "work": [("People", "Figures & dates", "every figure said (an amount, a percentage, a quantity, a date, a deadline), one a line, with what it refers to and the time it was said.")],
    "lesson": [("Key concepts", "Terms", "the terms introduced, each with what it means."),
               ("To review", "Homework & dates", "homework, readings, exams and their dates.")],
}
RECAP_RULES = [
    "- The Summary and the Topics carry the main threads by the weight the conversation gave them, not every aside.",
    "- Decisions are what was agreed; Action items what someone committed to do; People everyone who spoke or was named.",
    "- Notes the user took during the conversation are true: use them, and let them correct what the transcript seems to say.",
]


# ---------- the server ----------

def post(body):
    data = json.dumps({"model": MODEL, "stream": False, **body}).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode("utf-8", "replace")}, time.perf_counter() - t0
    return out, time.perf_counter() - t0


def usage_of(out):
    u = out.get("usage") or {}
    return {"prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion": u.get("completion_tokens")}


# ---------- the transcript, the cadence, the context ----------

MD_LINE = re.compile(r"^(?:\*\*(.+?)\*\* )?\((~?)(?:(\d+):)?(\d{1,2}):(\d{2})\): (.*)$")


def read_md(path):
    """A Transcribe `.md` export: `# title`, then `**Speaker** (mm:ss): text` a paragraph each.
    Returns (title, lines), each line `[mm:ss] Speaker: text` with the time in total minutes, as the page gives it."""
    title, lines = None, []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.startswith("# ") and title is None:
            title = raw[2:].strip()
            continue
        m = MD_LINE.match(raw.strip())
        if not m:
            continue
        speaker, approx, h, mm, ss, text = m.groups()
        minutes = int(h or 0) * 60 + int(mm)
        lines.append({"ms": (minutes * 60 + int(ss)) * 1000,
                      "line": f"[{approx}{minutes:02d}:{ss}] " + (f"{speaker}: " if speaker else "") + text})
    return title, lines


def rename(lines, me):
    if not me:
        return lines
    return [{**l, "line": l["line"].replace(f"] {me}: ", "] Me: ", 1)} for l in lines]


def triggers(lines, first=3, gap_ms=60_000):
    """Indices (exclusive end) at which a listening that keeps the notes up updates them, by transcript time."""
    out, last_ms, covered = [], None, 0
    for i in range(1, len(lines) + 1):
        now = lines[i - 1]["ms"]
        if covered == 0 and i - covered < first:
            continue
        if covered > 0 and now - last_ms < gap_ms:
            continue
        out.append(i)
        covered, last_ms = i, now
    if not out or out[-1] != len(lines):
        out.append(len(lines))
    return out


def analysis_points(lines, first=6, gap_ms=90_000, fresh_min=3):
    """Where the look runs: from the sixth utterance, then every 90 s while three new utterances came."""
    out, last_ms, analyzed = [], None, 0
    for i in range(1, len(lines) + 1):
        now = lines[i - 1]["ms"]
        if analyzed == 0 and i < first:
            continue
        if analyzed > 0 and (i - analyzed < fresh_min or now - last_ms < gap_ms):
            continue
        out.append(i)
        analyzed, last_ms = i, now
    return out


def text_of(lines):
    return "".join(l["line"] + "\n" for l in lines)


def context_block(write_in):
    """#440's block (web/src/transcribe/taskContext.ts), for a transcript whose context the user left empty."""
    return "\n".join([
        "Context of the conversation, told by the user:",
        f"- Write in: {write_in}.",
        "- Nothing else was told: no languages, terms or topic.",
        '- "Me" is the user, the person you work for. The other speakers are not named.',
        "- A sentence said as a hypothetical or an example is not a fact about the speakers.",
    ]) + "\n\n"


# ---------- notes as lines ----------

def parse_markdown(md, sections):
    """A full writing's answer as {section: [line, ...]}: a bullet (or a paragraph line) per line, `—` for none."""
    out, cur = {s: [] for s in sections}, None
    for raw in md.splitlines():
        h = re.match(r"^##\s+(.*?)\s*$", raw)
        if h:
            cur = h.group(1) if h.group(1) in out else None
            continue
        if cur is None:
            continue
        t = re.sub(r"^([-*+]|\d+[.)])\s+", "", raw.strip()).strip()
        if t and t != "—":
            out[cur].append(t)
    return out


def compare(prev, cur):
    """Lines kept word for word, added and removed, section by section (a changed line counts as one of each)."""
    kept = added = removed = 0
    for s in cur:
        a = list(prev.get(s, []))
        for line in cur[s]:
            if line in a:
                a.remove(line)
                kept += 1
            else:
                added += 1
        removed += len(a)
    return {"kept": kept, "added": added, "removed": removed}


class Running:
    """The running notes as the page keeps them: lines with ids never reused, a cap per section, the guards."""

    def __init__(self, kind):
        self.sections = [s for s, _, _ in RUNNING[kind]]
        self.cap = {s: c for s, c, _ in RUNNING[kind]}
        self.lines, self.order, self.next = {}, [], 1
        self.guards = {"ids_stripped": 0, "duplicate": 0, "overlong": 0}

    def of(self, s):
        return [i for i in self.order if self.lines[i]["section"] == s]

    def view(self):
        return {s: [self.lines[i]["text"] for i in self.of(s)] for s in self.sections}

    def clean(self, t):
        """Strips the ids of existing lines the model copied into a text."""
        t = (t or "").strip()
        ids = "|".join(re.escape(i) for i in self.lines) or "N0"
        c = re.sub(rf"^(?:[\(\[]?(?:{ids})[\)\]]?[:.)]?\s+)+", "", t)
        c = re.sub(rf"\s*[\(\[](?:{ids})[\)\]]", "", c).strip()
        if c != t:
            self.guards["ids_stripped"] += 1
        return c

    def check(self, s, t, but=None):
        if not t:
            return "Refused: the text is empty; remove_line removes a line."
        if len(t) > LINE_CHARS:
            self.guards["overlong"] += 1
            return f"Refused: the line is {len(t)} characters; a line holds {LINE_CHARS} at most. Write one short point."
        if any(self.lines[i]["text"].casefold() == t.casefold() for i in self.of(s) if i != but):
            self.guards["duplicate"] += 1
            return f"Refused: {s} already has that line."
        return None

    def apply(self, name, args):
        """One call: (True applied | False refused | None changed nothing, one-line result)."""
        if name == "add_line":
            s, t = args.get("section"), self.clean(args.get("text"))
            if s not in self.cap:
                return False, f"Refused: no section named {s!r}. Sections: {', '.join(self.sections)}."
            bad = self.check(s, t)
            if bad:
                return False, bad
            if len(self.of(s)) >= self.cap[s]:
                return False, f"Refused: {s} is full ({self.cap[s]} lines). Rewrite it with rewrite_section, merging what belongs together and folding this in."
            i = f"N{self.next}"
            self.next += 1
            self.lines[i] = {"section": s, "text": t}
            self.order.append(i)
            return True, f"Added {i} to {s}."
        if name in ("replace_line", "remove_line"):
            i = args.get("id")
            if i not in self.lines:
                return False, f"Refused: no line {i!r}."
            s = self.lines[i]["section"]
            if name == "remove_line":
                del self.lines[i]
                self.order.remove(i)
                return True, f"Removed {i}."
            t = self.clean(args.get("text"))
            if self.lines[i]["text"] == t:
                return None, f"Unchanged: {i} already reads so. When the notes are right, answer with no call."
            bad = self.check(s, t, but=i)
            if bad:
                return False, bad
            self.lines[i]["text"] = t
            return True, f"{i} updated."
        if name == "rewrite_section":
            s = args.get("section")
            if s not in self.cap:
                return False, f"Refused: no section named {s!r}."
            new, seen = [], set()
            for raw in args.get("lines") or []:
                t = self.clean(str(raw))
                if t and len(t) <= LINE_CHARS and t.casefold() not in seen:
                    seen.add(t.casefold())
                    new.append(t)
            if len(new) > self.cap[s]:
                return False, f"Refused: {len(new)} lines; {s} holds {self.cap[s]} at most. Send {self.cap[s]} or fewer: merge lines that belong together, or drop the least useful."
            old = self.of(s)
            if len(old) < self.cap[s]:
                return False, f"Refused: {s} has {len(old)} lines, not {self.cap[s]}: change single lines. rewrite_section is for a full section."
            by_text = {self.lines[i]["text"]: i for i in old}
            kept, ids = set(), []
            for t in new:
                if t in by_text and by_text[t] not in kept:
                    kept.add(by_text[t])
                    ids.append(by_text[t])
                else:
                    i = f"N{self.next}"
                    self.next += 1
                    self.lines[i] = {"section": s, "text": t}
                    ids.append(i)
            for i in old:
                if i not in kept:
                    del self.lines[i]
            self.order = [i for i in self.order if i in self.lines and self.lines[i]["section"] != s] + ids
            if len(kept) == len(old) == len(ids):
                return None, f"Unchanged: {s} already reads so. When the notes are right, answer with no call."
            return True, f"{s} rewritten: {len(kept)} kept, {len(ids) - len(kept)} new, {len(old) - len(kept)} removed."
        return False, f"Refused: no tool {name!r}."

    def snapshot(self):
        return "\n\n".join(f"{s}:\n" + ("\n".join(f"{i} {self.lines[i]['text']}" for i in self.of(s)) or "(empty)") for s in self.sections)

    def markdown(self):
        return "\n\n".join(f"## {s}\n" + ("\n".join(f"- {t}" for t in v) or "—") for s, v in self.view().items())


def running_task(kind):
    secs = "\n\n".join(f"## {s} ({c} line{'s' if c > 1 else ''} at most)\n{ask}." for s, c, ask in RUNNING[kind])
    rules = "\n".join([
        "Rules:",
        "- The notes live outside this conversation, one line at a time, and change only through the tools: add_line adds a line to a section, replace_line rewrites a line by its id, remove_line removes one, rewrite_section rewrites a whole section. A tool's result says what it did.",
        "- Each user turn brings what was said since the last one. Change only what it makes wrong, stale or incomplete. A line still true stays as it is, word for word. When nothing changes, call no tool.",
        "- Keep the notes short: a section holds at most the lines given above. Change single lines; only when a section is full, rewrite it with rewrite_section: merge, drop what is stale, fold in what is new; a line kept word for word keeps its id.",
        "- A line is one short point; a figure gets a line of its own. Never write a line's id in its text.",
        "- Write the lines in the language the context says to write in.",
        "- Point at a moment by its time, as [mm:ss], in minutes from the start: [64:09] is sixty-four minutes in.",
        "- A short phrase repeated word for word, again and again, is speech-recognition noise: never note it.",
        "- Make every change the turn calls for, one tool call each: several calls in one answer are fine. When the results come back, make any change still missing, or answer with no call once the notes are right.",
        "- Answer only with tool calls, never with text.",
    ])
    return f"Keep the running notes of {P['kinds'][kind]['about']}: notes that help the user follow it and take part while it goes on. These sections, in this order:\n\n{secs}\n\n{rules}"


def running_tools(kind):
    sections = [s for s, _, _ in RUNNING[kind]]
    return [
        {"type": "function", "function": {
            "name": "add_line", "description": "Adds one line at the end of a section of the notes.",
            "parameters": {"type": "object", "properties": {
                "section": {"type": "string", "enum": sections},
                "text": {"type": "string", "description": "The line, without a bullet marker."}},
                "required": ["section", "text"]}}},
        {"type": "function", "function": {
            "name": "replace_line", "description": "Rewrites one line of the notes, by its id.",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string", "description": "The line's id, such as N7."},
                "text": {"type": "string"}},
                "required": ["id", "text"]}}},
        {"type": "function", "function": {
            "name": "remove_line", "description": "Removes one line of the notes, by its id.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
        {"type": "function", "function": {
            "name": "rewrite_section",
            "description": "Rewrites a whole section of the notes, only once it is full: these lines, at most the section's limit, take the place of the ones it has. A line kept word for word keeps its id.",
            "parameters": {"type": "object", "properties": {
                "section": {"type": "string", "enum": sections},
                "lines": {"type": "array", "items": {"type": "string"}}},
                "required": ["section", "lines"]}}},
    ]


# ---------- the three ways ----------

def looks(title, ctx, lines, due, rows):
    for a in due:
        body = {**ANALYSIS_SETTINGS, "messages": [{"role": "system", "content": P["analysis"]["system"]},
                                                  {"role": "user", "content": ctx + f"{title}\n\n" + text_of(lines[:a])}]}
        out, secs = post(body)
        rows.append({"way": "look", "covered": a, "secs": round(secs, 2), **(usage_of(out) if "error" not in out else {"error": out})})


def run_r(title, ctx, lines, kind, points, apoints, outdir, restart_tokens):
    notes = Running(kind)
    system = {"role": "system", "content": ASSIST + "\n\n" + running_task(kind)}
    tools = running_tools(kind)
    rows, log, snaps = [], [], []
    messages, start, done_a, last_prompt, restarts = None, 0, 0, 0, 0
    for idx, n in enumerate(points):
        looks(title, ctx, lines, [a for a in apoints if done_a < a <= n], rows)
        done_a = n
        new = lines[start:n]
        end = "\nThe conversation is over." if idx == len(points) - 1 else ""
        fresh = messages is None or last_prompt > restart_tokens
        if fresh:
            restarts += messages is not None
            state = "The notes are empty." if not notes.order else "The notes so far, each line with its id:\n\n" + notes.snapshot()
            messages = [system, {"role": "user", "content": ctx + f"{title}\n\n" + text_of(lines[:n]) + f"\n{state}{end}"}]
        else:
            stamp = new[0]["line"][1:].split("]")[0].lstrip("~")
            messages.append({"role": "user", "content": f"Said since [{stamp}]:\n" + text_of(new) + end})
        prev = notes.view()
        agg = {"secs": 0.0, "completion": 0, "calls": 0, "rounds": 0, "noop": 0, "refused": 0, "bad_json": 0, "text_instead": 0,
               "cached0": None, "prompt0": None, "capped": False, "restart": fresh and idx > 0, "refusals": []}
        g0 = dict(notes.guards)
        changes, idle = [], 0
        for rnd in range(MAX_ROUNDS):
            out, secs = post({**SETTINGS, "messages": messages, "tools": tools, "tool_choice": "auto"})
            if "error" in out:
                raise SystemExit(f"server error at {n}: {out}")
            u = usage_of(out)
            last_prompt = u["prompt"] or last_prompt
            if rnd == 0:
                agg["cached0"], agg["prompt0"] = u["cached"], u["prompt"]
            msg = out["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            agg["secs"] += secs
            agg["completion"] += u["completion"] or 0
            agg["rounds"] += 1
            agg["calls"] += len(calls)
            if not calls and (msg.get("content") or "").strip():
                agg["text_instead"] += 1
            pending, noops, refused_now = [], 0, 0
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    agg["bad_json"] += 1
                    refused_now += 1
                    pending.append({"role": "tool", "tool_call_id": c["id"], "content": "Refused: the arguments are not JSON."})
                    continue
                sec = args.get("section") if name in ("add_line", "rewrite_section") else notes.lines.get(args.get("id"), {}).get("section")
                before = notes.view().get(sec, [])
                old = notes.lines.get(args.get("id"), {}).get("text")
                good, result = notes.apply(name, args)
                if good is None:
                    agg["noop"] += 1
                    noops += 1
                elif not good:
                    agg["refused"] += 1
                    refused_now += 1
                    agg["refusals"].append(result)
                else:
                    after = notes.view().get(sec, [])
                    changes.append({"op": name, "section": sec, "old": old if name != "rewrite_section" else [t for t in before if t not in after],
                                    "new": (args.get("text") or "").strip() if name != "rewrite_section" else [t for t in after if t not in before]})
                pending.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            echo = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                echo["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in calls]
            messages.append(echo)
            if not calls:
                break
            messages += pending
            # A round whose calls all changed nothing (unchanged or refused) is idle; two in a row end the update.
            idle = idle + 1 if noops + refused_now == len(calls) else 0
            if idle >= 2:
                break
            if rnd == MAX_ROUNDS - 1:
                agg["capped"] = True
        cur = notes.view()
        at = lines[n - 1]["line"][1:].split("]")[0].lstrip("~")
        agg["secs"] = round(agg["secs"], 2)
        rows.append({"way": "R", "covered": n, "at": at, **agg, **compare(prev, cur),
                     "guards": {k: notes.guards[k] - g0[k] for k in g0},
                     "lines": len(notes.order), "per_section": {s: len(v) for s, v in cur.items()}, "thread_tokens": last_prompt})
        snaps.append({"covered": n, "at": at, "notes": cur})
        log.append((new[0]["line"][1:].split("]")[0].lstrip("~") if new else at, at, changes, fresh and idx > 0))
        start = n
        print(f"R {idx + 1}/{len(points)} at {at}: {agg['rounds']} rounds, {agg['secs']} s", flush=True)
    md = [f"# Running notes, {len(points)} updates, {restarts} restarts", ""]
    for frm, to, changes, restarted in log:
        md.append(f"**[{frm} → {to}]**" + (" *(restarted from the snapshot)*" if restarted else "") + ("" if changes else " no change"))
        for c in changes:
            if c["op"] == "rewrite_section":
                md.append(f"- rewrite *{c['section']}*: " + "; ".join(f"~~{t}~~" for t in c["old"]) + " → " + ("; ".join(c["new"]) or "—"))
            elif c["op"] == "add_line":
                md.append(f"- add *{c['section']}*: {c['new']}")
            elif c["op"] == "remove_line":
                md.append(f"- remove *{c['section']}*: ~~{c['old']}~~")
            else:
                md.append(f"- edit *{c['section']}*: ~~{c['old']}~~ → {c['new']}")
        md.append("")
    md += ["## At the end", "", notes.markdown()]
    (outdir / "running.md").write_bytes("\n".join(md).encode("utf-8"))
    (outdir / "running.snapshots.json").write_bytes(json.dumps(snaps, ensure_ascii=False, indent=1).encode("utf-8"))
    return rows


def run_a(title, ctx, lines, kind, points, apoints, outdir):
    system = ASSIST + "\n\n" + P["kinds"][kind]["task"].replace(LANG_OLD, LANG_NEW)
    sections = [s["name"] for s in P["kinds"][kind]["sections"]]
    rows, prev, final, done_a = [], {s: [] for s in sections}, "", 0
    for idx, n in enumerate(points):
        looks(title, ctx, lines, [a for a in apoints if done_a < a <= n], rows)
        done_a = n
        out, secs = post({**SETTINGS, "messages": [{"role": "system", "content": system},
                                                   {"role": "user", "content": ctx + f"{title}\n\n" + text_of(lines[:n])}]})
        if "error" in out:
            raise SystemExit(f"server error at {n}: {out}")
        md = out["choices"][0]["message"].get("content") or ""
        cur = parse_markdown(md, sections)
        rows.append({"way": "A", "covered": n, "secs": round(secs, 2), **usage_of(out), **compare(prev, cur),
                     "lines": sum(len(v) for v in cur.values())})
        prev, final = cur, md
        print(f"A {idx + 1}/{len(points)}: {round(secs, 1)} s", flush=True)
    (outdir / "full-rewrite.md").write_bytes(final.encode("utf-8"))
    return rows


def recap_task(kind):
    task = P["kinds"][kind]["task"].replace(LANG_OLD, LANG_NEW)
    for after, name, ask in RECAP_EXTRA.get(kind, []):
        m = re.search(rf"## {re.escape(after)}\n[^\n]*\n", task)
        task = task[:m.end()] + f"\n## {name}\n{ask}\n" + task[m.end():]
    task = task.replace("Write the notes of", "Write the recap of", 1)
    return task.replace("\nRules:\n", "\nRules:\n" + "\n".join(RECAP_RULES) + "\n", 1)


def run_c(title, ctx, lines, kind, outdir, user_lines=()):
    took = "Notes the user took during the conversation: none." if not user_lines else \
        "Notes the user took during the conversation, to be taken as true:\n" + "\n".join(f"- {t}" for t in user_lines)
    body = {**SETTINGS, "max_tokens": 3000, "messages": [{"role": "system", "content": ASSIST + "\n\n" + recap_task(kind)},
                                                         {"role": "user", "content": ctx + f"{title}\n\n" + text_of(lines) + f"\n{took}"}]}
    out, secs = post(body)
    if "error" in out:
        raise SystemExit(f"server error: {out}")
    (outdir / "recap.md").write_bytes((out["choices"][0]["message"].get("content") or "").encode("utf-8"))
    print(f"C: {round(secs, 1)} s", flush=True)
    return [{"way": "C", "covered": len(lines), "secs": round(secs, 2), **usage_of(out)}]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("md")
    ap.add_argument("kind", choices=list(RUNNING))
    ap.add_argument("label")
    ap.add_argument("--me", default=None, help="the speaker label to rename Me, the user")
    ap.add_argument("--write-in", default="English")
    ap.add_argument("--ways", default="R,A,C")
    ap.add_argument("--restart-tokens", type=int, default=64_000, help="the running notes restart from a snapshot past this prompt")
    ap.add_argument("--out", default=str(HERE.parent.parent / ".scratch" / "notes-replay"))
    a = ap.parse_args()
    title, lines = read_md(a.md)
    title = title or a.label
    lines = rename(lines, a.me)
    ctx = context_block(a.write_in)
    outdir = Path(a.out) / a.label
    outdir.mkdir(parents=True, exist_ok=True)
    points, apoints = triggers(lines), analysis_points(lines)
    meta = {"label": a.label, "kind": a.kind, "me": a.me, "utterances": len(lines), "minutes": round(lines[-1]["ms"] / 60000, 1),
            "triggers": len(points), "looks": len(apoints), "model": MODEL, "restart_tokens": a.restart_tokens,
            "started": time.strftime("%Y-%m-%d %H:%M")}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(meta, flush=True)
    for way in a.ways.split(","):
        if way == "R":
            rows = run_r(title, ctx, lines, a.kind, points, apoints, outdir, a.restart_tokens)
        elif way == "A":
            rows = run_a(title, ctx, lines, a.kind, points, apoints, outdir)
        else:
            rows = run_c(title, ctx, lines, a.kind, outdir)
        with open(outdir / f"rows.{way}.jsonl", "wb") as f:
            for r in rows:
                f.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
