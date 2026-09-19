"""Mask out comments and string literals, preserving byte offsets.

Every symbol-extraction rule downstream (brace matching, `fn NAME`,
`const NAME =`) is a lexical rule that must not fire inside a comment or a
string.  Instead of teaching each rule about quoting, this module produces a
*masked* copy of the source: same length, same offsets, with every comment and
string-literal character replaced by a space.  Downstream code reads the mask
and indexes back into the original text for the actual names.

The three languages differ only in which quoting forms exist, so one state
machine with a per-language flag covers all of them.
"""

RUST = "rust"
C = "c"
TS = "ts"

EXT_LANG = {
    ".rs": RUST,
    ".cu": C, ".cuh": C, ".h": C, ".hpp": C, ".cpp": C, ".cc": C,
    ".ts": TS, ".tsx": TS,
}

_IDENT_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$"
)

_REGEX_PRECEDING_KEYWORDS = {
    "return", "typeof", "instanceof", "in", "of", "new", "delete",
    "void", "throw", "case", "do", "else", "yield", "await",
}


def mask_source(src, lang):
    """Return (masked source, list of lexer complaints).

    Complaints are never fatal: an unterminated block comment at EOF is a real
    thing in generated files, and the caller counts them rather than trusting a
    silent result.
    """
    out = list(src)
    problems = []
    n = len(src)
    i = 0
    # TS template literals nest through `${ }` holes; the stack records the
    # brace depth at which each open template resumes.
    tpl_stack = []
    brace_depth = 0

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]

        if tpl_stack and c == "}" and brace_depth == tpl_stack[-1]:
            # closing a `${ ... }` hole: back inside the template text
            tpl_stack.pop()
            out[i] = " "
            i += 1
            j = _ts_template_body(src, i, out, tpl_stack, brace_depth)
            if j is None:
                problems.append("unterminated template literal at %d" % i)
                blank(i, n)
                break
            i = j
            continue

        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue

        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j, ok = _block_comment_end(src, i, lang)
            if not ok:
                problems.append("unterminated block comment at %d" % i)
            blank(i, j)
            i = j
            continue

        if lang == RUST and c in "rb" and _rust_raw_start(src, i):
            j = _rust_raw_end(src, i)
            if j is None:
                problems.append("unterminated raw string at %d" % i)
                blank(i, n)
                break
            blank(i, j)
            i = j
            continue

        if c == '"':
            # Only Rust lets a `"` string carry a raw newline.  In C and TS a
            # quote with no closer on its line is not a string at all — it is
            # JSX text (`<p>say "hi"</p>` split over lines) or a stray byte —
            # so it stays as ordinary source instead of swallowing the file.
            j = _quoted_end(src, i, '"', multiline=(lang == RUST))
            if j is None:
                if lang == RUST:
                    problems.append("unterminated string at %d" % i)
                    blank(i, n)
                    break
                i += 1
                continue
            blank(i, j)
            i = j
            continue

        if c == "'":
            if lang == RUST:
                j = _rust_char_literal_end(src, i)
                if j is None:
                    i += 1      # a lifetime: ordinary code, leave it alone
                    continue
            elif lang == C and _c_digit_separator(src, i):
                i += 1          # `10'000`, `65'535`: part of the number
                continue
            else:
                j = _quoted_end(src, i, "'", multiline=False)
                if j is None:
                    # an apostrophe in JSX text (`this session's log`)
                    i += 1
                    continue
            blank(i, j)
            i = j
            continue

        if lang == TS and c == "`":
            out[i] = " "
            i += 1
            j = _ts_template_body(src, i, out, tpl_stack, brace_depth)
            if j is None:
                problems.append("unterminated template literal at %d" % i)
                blank(i, n)
                break
            i = j
            continue

        if lang == TS and c == "/" and _regex_can_start(out, i):
            j = _regex_end(src, i)
            if j is not None:
                blank(i, j)
                i = j
                continue

        if c == "{":
            brace_depth += 1
        elif c == "}":
            brace_depth -= 1
        i += 1

    return "".join(out), problems


def _block_comment_end(src, i, lang):
    """Index just past the closing delimiter, and whether one was found.
    Rust nests `/* */`; C and TS do not."""
    n = len(src)
    if lang == RUST:
        depth = 0
        j = i
        while j < n - 1:
            if src[j] == "/" and src[j + 1] == "*":
                depth += 1
                j += 2
            elif src[j] == "*" and src[j + 1] == "/":
                depth -= 1
                j += 2
                if depth == 0:
                    return j, True
            else:
                j += 1
        return n, False
    j = src.find("*/", i + 2)
    return (n, False) if j < 0 else (j + 2, True)


def _rust_char_literal_end(src, i):
    """`'` in Rust opens a char literal only when a closing `'` follows within
    the span of one character; otherwise it opens a lifetime (`'a`, `'static`).
    Returns the index just past the closing quote, or None for a lifetime."""
    n = len(src)
    j = i + 1
    if j >= n:
        return None
    if src[j] == "\\":
        j += 1
        while j < n and src[j] != "'":
            # `\u{1F600}` and friends, bounded so a lifetime never swallows a line
            if src[j] == "\n" or j - i > 12:
                return None
            j += 1
        return j + 1 if j < n else None
    if j + 1 < n and src[j + 1] == "'":
        return j + 2
    return None


def _quoted_end(src, i, quote, multiline):
    """Index just past the closing quote of a simple escaped literal, or None
    when there is no closer (before the end of the line unless `multiline`)."""
    j = i + 1
    n = len(src)
    while j < n:
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == quote:
            return j + 1
        if src[j] == "\n" and not multiline:
            return None
        j += 1
    return None


def _c_digit_separator(src, i):
    """C++14 lets `'` separate digit groups: `10'000`, `65'535`, `0xFF'FF`."""
    return (i > 0 and i + 1 < len(src)
            and src[i - 1] in "0123456789abcdefABCDEF"
            and src[i + 1] in "0123456789abcdefABCDEF")


def _rust_raw_start(src, i):
    n = len(src)
    j = i
    if src[j] == "b":
        j += 1
    if j >= n or src[j] != "r":
        return False
    j += 1
    while j < n and src[j] == "#":
        j += 1
    return j < n and src[j] == '"'


def _rust_raw_end(src, i):
    j = i
    if src[j] == "b":
        j += 1
    j += 1                                    # the `r`
    hashes = 0
    while j < len(src) and src[j] == "#":
        hashes += 1
        j += 1
    closer = '"' + "#" * hashes
    k = src.find(closer, j + 1)
    return None if k < 0 else k + len(closer)


def _regex_can_start(masked, i):
    """A `/` in TS opens a regex literal only where a value cannot precede it.
    Scanning back over the already-masked output keeps comments and strings from
    voting.  This is a heuristic: `a = b / c / d` is not decidable lexically."""
    j = i - 1
    while j >= 0 and masked[j] in " \t\r\n":
        j -= 1
    if j < 0:
        return True
    c = masked[j]
    if c in ")]":
        return False
    if c in _IDENT_CHARS:
        # keywords (`return`, `typeof`, ...) may be followed by a regex;
        # bare identifiers and numbers may not.
        k = j
        while k >= 0 and masked[k] in _IDENT_CHARS:
            k -= 1
        return "".join(masked[k + 1:j + 1]) in _REGEX_PRECEDING_KEYWORDS
    return True


def _regex_end(src, i):
    """A regex literal ends at the first unescaped `/` outside a character class
    and before a newline; anything else is division."""
    j = i + 1
    n = len(src)
    in_class = False
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            return None
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            j += 1
            while j < n and src[j].isalpha():
                j += 1                        # flags
            return j
        j += 1
    return None


def _ts_template_body(src, i, out, tpl_stack, brace_depth):
    """Blank template text from `i` until either the closing backtick (returns
    the index past it) or a `${` hole, which pushes onto `tpl_stack` and hands
    control back to the main loop so the hole is lexed as code."""
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            out[i] = " "
            if i + 1 < n and out[i + 1] != "\n":
                out[i + 1] = " "
            i += 2
            continue
        if c == "`":
            out[i] = " "
            return i + 1
        if c == "$" and i + 1 < n and src[i + 1] == "{":
            # Both `$` and `{` are template punctuation, not code: blanking the
            # brace too is what keeps the masked source brace-balanced.  The
            # hole's own braces balance among themselves, so the `}` that closes
            # it surfaces at exactly the depth recorded here.
            out[i] = " "
            out[i + 1] = " "
            tpl_stack.append(brace_depth)
            return i + 2
        if c != "\n":
            out[i] = " "
        i += 1
    return None
