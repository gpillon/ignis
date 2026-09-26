"""What the three families share: the word rules behind the lexical /
paraphrase split, and the question rows every family writes.

The split is spec 18's guard against ICR's lexical bias
(`docs/specs/decide/18-locate-by-attention.md` § Phase A): a **lexical**
question shares a rare word with its target -- one that appears in the target
and in no other segment -- and a **paraphrase** question shares no content
word with it at all. Both are checked here on every generated question rather
than trusted to the banks, so a bank entry that leaks a word is caught when a
set is written, not after it was measured.

"Word" is deliberately crude: lowercase runs of letters and digits, compared
by their first five characters, so `timed` and `timeout` are one word and a
paraphrase cannot pass by changing a suffix. The frame words every question
of a family uses ("which line reports ...") are not content.
"""

import re
import unicodedata

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had
has have how if in into is it its no not of on or so than that the their them
then there these they this those to too was were what when where which while
who whom why will with would you your our we he she his her him after before
about above again against all any because both each few more most other over
own same some such under until very just now only out off up down also once
here more why one two its

line lines log logs entry entries record records item items employee employees
ticket tickets product products sentence sentences reports report says say
mentions mention shows show states state tells tell list evidence question
answer helps help best supports support find
""".split())

WORD = re.compile(r"[a-z0-9]+")


def nfc(text):
    """The tokenizer normalizes to NFC, so the sets are written in it: the
    bytes the harness records a segment by are the bytes the model reads."""
    return unicodedata.normalize("NFC", text)


def words(text):
    return WORD.findall(text.lower())


def stem(word):
    return word[:5]


def content_stems(text):
    """The stems of `text`'s content words: not a stopword, and at least three
    characters or a number."""
    return {stem(w) for w in words(text)
            if w not in STOPWORDS and (len(w) >= 3 or w.isdigit())}


def rare_shared(question, target, others):
    """Content stems the question shares with the target and with no other
    segment -- what makes a question **lexical**."""
    shared = content_stems(question) & content_stems(target)
    for other in others:
        if not shared:
            break
        shared -= content_stems(other)
    return shared


def paraphrase_clean(question, target):
    """True when the question shares no content stem with the target."""
    return not (content_stems(question) & content_stems(target))


def assign_absent(rng, count):
    """Which of `count` questions have their target removed: one in six,
    rounded, chosen by the set's own generator so it is deterministic per
    seed and spread over lengths and splits rather than tied to either."""
    order = list(range(count))
    rng.shuffle(order)
    return set(order[:round(count / 6)])
