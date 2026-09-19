"""Find symbol definitions and the extent of their bodies.

The index this study builds is keyed by symbol name (`02-storage.md` §5.1), and
both poolings need the body extent: "last token of the definition" is its end,
"mean over the body" is its whole span.  So the unit produced here is
`(file, kind, name, name_span, body_span)` where the spans are character
offsets into the *original* source.

Detection runs over the masked source from `lexmask`, so nothing fires inside a
comment or a string.  Names are then read back out of the original text (the
mask preserves offsets, and identifiers are never masked anyway).

Parse failures are counted, never swallowed: a definition whose body brace does
not close is reported so the caller can see how much of the corpus the parser
does not understand.
"""

import os
import re

from lexmask import EXT_LANG, RUST, C, TS, mask_source

CORPUS_EXTS = tuple(EXT_LANG)
EXCLUDED_DIRS = {"target", "node_modules", ".git", "build", "dist", ".scratch",
                 "__pycache__", ".venv", "venv"}


class Symbol:
    __slots__ = ("path", "lang", "kind", "name", "name_span", "body_span",
                 "depth")

    def __init__(self, path, lang, kind, name, name_span, body_span, depth=0):
        self.path = path
        self.lang = lang
        self.kind = kind
        self.name = name
        self.name_span = name_span      # (start, end) of the name itself
        self.body_span = body_span      # (start, end) of the whole definition
        self.depth = depth              # brace nesting at the definition site

    def key(self):
        return "%s::%s::%s" % (self.path, self.kind, self.name)

    def as_dict(self):
        return {
            "path": self.path, "lang": self.lang, "kind": self.kind,
            "name": self.name, "depth": self.depth,
            "name_start": self.name_span[0], "name_end": self.name_span[1],
            "body_start": self.body_span[0], "body_end": self.body_span[1],
        }

    def __repr__(self):
        return "Symbol(%s %s %r %s)" % (self.lang, self.kind, self.name,
                                        self.body_span)


# `02-storage.md` §4 counted Rust symbols with exactly these keywords.
_RUST_BRACED = re.compile(
    r"\b(fn|struct|enum|trait|union)\s+([A-Za-z_][A-Za-z0-9_]*)")
# A Rust item, not a type: `*const u8` and `&'static str` also read as
# "const IDENT" / "static IDENT", so the declaration syntax is required —
# `const NAME:` and `static NAME:` always carry the type annotation, and
# `type NAME` is followed by `=`, `;` or a generic parameter list.  The
# lookbehind rules out the pointer and lifetime forms outright.
_RUST_TERMINATED = re.compile(
    r"(?<![*'\w])\b(?:(const|static)\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:"
    r"|(type)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?=[=;<]))")

_C_BRACED = re.compile(
    r"\b(struct|enum|union|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:final\s*)?"
    r"(?::[^;{]*)?\{")
# A C/CUDA function definition: a name, a parenthesised list, then `{` — with
# only qualifiers allowed in between, which is what rules out a call site.
_C_FUNC = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_C_TYPEDEF = re.compile(r"\btypedef\b")

_TS_BRACED = re.compile(
    r"\b(class|interface|enum|namespace)\s+([A-Za-z_$][A-Za-z0-9_$]*)")
_TS_FUNC = re.compile(
    r"\bfunction\s*\*?\s+([A-Za-z_$][A-Za-z0-9_$]*)")
# `let` and `var` are excluded on purpose: they are re-assignable locals, never
# a definition worth keying an index on (`02-storage.md` §5.2).
_TS_BINDING = re.compile(
    r"\b(const|type)\s+([A-Za-z_$][A-Za-z0-9_$]*)")

_C_NON_FUNC_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "do",
    "else", "defined", "static_assert", "assert", "__launch_bounds__",
    "alignas", "decltype", "noexcept", "operator", "constexpr", "and", "or",
    "not", "typeof", "typeid", "throw", "new", "delete", "case",
}


def iter_corpus(root, exts=CORPUS_EXTS, excluded=EXCLUDED_DIRS):
    """Yield repo-relative paths of every corpus file under `root`, in a stable
    order — the index must be reproducible byte for byte (`02-storage.md` §7.1)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded)
        for fn in sorted(filenames):
            if fn.endswith(exts):
                full = os.path.join(dirpath, fn)
                out.append(os.path.relpath(full, root).replace("\\", "/"))
    out.sort()
    return out


def lang_of(path):
    return EXT_LANG.get(os.path.splitext(path)[1])


def _match_brace(masked, open_idx):
    """Index just past the `}` matching the `{` at `open_idx`, or None."""
    depth = 0
    for k in range(open_idx, len(masked)):
        c = masked[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return None


def _next_brace_or_semi(masked, start, stop_at_paren=True):
    """Walk forward from `start` to the `{` that opens the body, skipping
    balanced `(...)`, `[...]` and `<...>` so generics and argument lists do not
    hide it.  Returns ('{', idx), (';', idx) or (None, None)."""
    depth_paren = 0
    depth_angle = 0
    k = start
    n = len(masked)
    while k < n:
        c = masked[k]
        if c in "([":
            depth_paren += 1
        elif c in ")]":
            depth_paren -= 1
            if depth_paren < 0:
                return None, None
        elif c == "<":
            depth_angle += 1
        elif c == ">":
            if depth_angle > 0:
                depth_angle -= 1
        elif depth_paren == 0:
            if c == "{":
                return "{", k
            if c == ";":
                return ";", k
            if c == "=" and stop_at_paren and depth_angle == 0:
                # `const X: T = ...;` — the body is the initialiser
                pass
        k += 1
    return None, None


def _rust_symbols(src, masked, path, problems):
    found = []
    for m in _RUST_BRACED.finditer(masked):
        kind, name = m.group(1), m.group(2)
        what, idx = _next_brace_or_semi(masked, m.end(2))
        if what == "{":
            end = _match_brace(masked, idx)
            if end is None:
                problems.append("%s: unclosed body for %s %s" % (path, kind, name))
                continue
        elif what == ";":
            end = idx + 1                      # `struct Foo;`, `fn f();`
        else:
            problems.append("%s: no body for %s %s" % (path, kind, name))
            continue
        found.append(Symbol(path, RUST, kind, name, m.span(2), (m.start(1), end)))
    for m in _RUST_TERMINATED.finditer(masked):
        kind = m.group(1) or m.group(3)
        name_span = m.span(2) if m.group(1) else m.span(4)
        name = masked[name_span[0]:name_span[1]]
        what, idx = _next_brace_or_semi(masked, name_span[1])
        if what == "{":
            # `const X: T = Foo { .. };` — take the statement, not just the block
            end = _match_brace(masked, idx)
            if end is None:
                problems.append("%s: unclosed init for %s %s" % (path, kind, name))
                continue
            semi = masked.find(";", end)
            end = end if semi < 0 else semi + 1
        elif what == ";":
            end = idx + 1
        else:
            problems.append("%s: unterminated %s %s" % (path, kind, name))
            continue
        found.append(Symbol(path, RUST, kind, name, name_span, (m.start(), end)))
    return found


def _c_symbols(src, masked, path, problems):
    found = []
    for m in _C_BRACED.finditer(masked):
        kind, name = m.group(1), m.group(2)
        open_idx = masked.index("{", m.end(2))
        end = _match_brace(masked, open_idx)
        if end is None:
            problems.append("%s: unclosed body for %s %s" % (path, kind, name))
            continue
        semi = masked.find(";", end)
        if semi >= 0 and masked[end:semi].strip() == "":
            end = semi + 1
        found.append(Symbol(path, C, kind, name, m.span(2), (m.start(1), end)))

    for m in _C_FUNC.finditer(masked):
        name = m.group(1)
        if name in _C_NON_FUNC_KEYWORDS:
            continue
        close = _matching_paren(masked, m.end(1))
        if close is None:
            continue
        k = close
        n = len(masked)
        while k < n and masked[k] in " \t\r\n":
            k += 1
        # qualifiers between `)` and `{`: const, noexcept, override, -> T, : init
        while k < n and masked[k] not in "{;=)":
            k += 1
            while k < n and masked[k] in " \t\r\n":
                k += 1
        if k >= n or masked[k] != "{":
            continue
        line_start = masked.rfind("\n", 0, m.start(1)) + 1
        if masked[line_start:m.start(1)].strip().endswith(("=", "return", ",")):
            continue
        end = _match_brace(masked, k)
        if end is None:
            problems.append("%s: unclosed body for fn %s" % (path, name))
            continue
        found.append(Symbol(path, C, "fn", name, m.span(1), (line_start, end)))

    for m in _C_TYPEDEF.finditer(masked):
        semi = masked.find(";", m.end())
        if semi < 0:
            problems.append("%s: unterminated typedef" % path)
            continue
        brace = masked.find("{", m.end())
        if 0 <= brace < semi:
            end_brace = _match_brace(masked, brace)
            if end_brace is None:
                problems.append("%s: unclosed typedef body" % path)
                continue
            semi = masked.find(";", end_brace)
            if semi < 0:
                continue
        names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*$",
                           masked[max(m.end(), semi - 80):semi])
        if not names:
            continue
        name = names[-1]
        start = masked.rindex(name, max(m.end(), semi - 80), semi)
        found.append(Symbol(path, C, "typedef", name, (start, start + len(name)),
                            (m.start(), semi + 1)))
    return found


def _matching_paren(masked, open_search_from):
    k = masked.index("(", open_search_from)
    depth = 0
    for j in range(k, len(masked)):
        if masked[j] == "(":
            depth += 1
        elif masked[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _ts_symbols(src, masked, path, problems):
    found = []
    for m in _TS_BRACED.finditer(masked):
        kind, name = m.group(1), m.group(2)
        what, idx = _next_brace_or_semi(masked, m.end(2))
        if what != "{":
            problems.append("%s: no body for %s %s" % (path, kind, name))
            continue
        end = _match_brace(masked, idx)
        if end is None:
            problems.append("%s: unclosed body for %s %s" % (path, kind, name))
            continue
        found.append(Symbol(path, TS, kind, name, m.span(2), (m.start(1), end)))

    for m in _TS_FUNC.finditer(masked):
        name = m.group(1)
        what, idx = _next_brace_or_semi(masked, m.end(1))
        if what != "{":
            continue
        end = _match_brace(masked, idx)
        if end is None:
            problems.append("%s: unclosed body for function %s" % (path, name))
            continue
        found.append(Symbol(path, TS, "function", name, m.span(1),
                            (m.start(), end)))

    for m in _TS_BINDING.finditer(masked):
        kind, name = m.group(1), m.group(2)
        end = _statement_end(masked, m.end(2))
        if end is None:
            problems.append("%s: unterminated %s %s" % (path, kind, name))
            continue
        found.append(Symbol(path, TS, kind, name, m.span(2), (m.start(1), end)))
    return found


def _statement_end(masked, start):
    """End of a TS binding: the `;` or newline at brace/paren depth zero,
    with `{...}`, `(...)` and `[...]` skipped as units."""
    depth = 0
    n = len(masked)
    k = start
    while k < n:
        c = masked[k]
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
            if depth < 0:
                return k
        elif depth == 0:
            if c == ";":
                return k + 1
            if c == "\n" and k > start:
                tail = masked[masked.rfind("\n", 0, k) + 1:k].rstrip()
                if tail and tail[-1] not in "=,+-*/|&?:(<[{":
                    return k
        k += 1
    return n


_DISPATCH = {RUST: _rust_symbols, C: _c_symbols, TS: _ts_symbols}


def parse_file(path, text):
    """Return (symbols, problems) for one corpus file."""
    lang = lang_of(path)
    if lang is None:
        return [], ["%s: unknown extension" % path]
    masked, problems = mask_source(text, lang)
    problems = ["%s: %s" % (path, p) for p in problems]
    syms = _DISPATCH[lang](text, masked, path, problems)
    depths = _depth_prefix(masked)
    for s in syms:
        s.name = text[s.name_span[0]:s.name_span[1]]
        s.depth = depths[s.body_span[0]]
    syms.sort(key=lambda s: (s.body_span[0], s.name))
    return syms, problems


def _depth_prefix(masked):
    """Brace nesting depth at every offset, so a symbol can say whether it is a
    file-level definition (depth 0) or nested inside another one."""
    out = [0] * (len(masked) + 1)
    d = 0
    for i, c in enumerate(masked):
        out[i] = d
        if c == "{":
            d += 1
        elif c == "}":
            d = max(0, d - 1)
    out[len(masked)] = d
    return out


def parse_corpus(root, paths=None):
    """Parse the whole corpus.  Returns (symbols, problems, per-file counts)."""
    paths = iter_corpus(root) if paths is None else paths
    all_syms, all_problems, per_file = [], [], {}
    for rel in paths:
        full = os.path.join(root, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            all_problems.append("%s: %s" % (rel, exc))
            continue
        syms, problems = parse_file(rel, text)
        all_syms.extend(syms)
        all_problems.extend(problems)
        per_file[rel] = len(syms)
    return all_syms, all_problems, per_file
