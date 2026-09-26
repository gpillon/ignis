"""Synthetic service logs with one line to find (spec 18, family `logs`).

A log is 60, 250 or 1,000 lines of `HH:MM LEVEL service: message`. One
ERROR line is the **target**, among 3 to 8 **distractor** ERROR lines of the
same shape drawn from the same bank of events; the rest is routine INFO,
DEBUG and WARN traffic. The question asks for the target's event:

- **lexical** -- it repeats a word the target carries and no other line does;
- **paraphrase** -- it describes the event without any of the target's
  content words, which a string matcher cannot follow;
- **absent** -- one question in six has its target replaced by a routine line.

Every event carries both phrasings, and `common.py` checks each generated
question against its own log, so a bank entry that leaks a word or loses its
rare one fails the generator instead of the measurement.
"""

from common import assign_absent, paraphrase_clean, rare_shared

LENGTHS = (60, 250, 1000)
SERVICES = ("gateway", "auth", "billing", "orders", "catalog", "search", "mailer",
            "jobs", "media", "ledger", "profile", "checkout", "edge", "reports-api")
HOSTS = ("db-primary", "db-replica-2", "cache-3", "api.partner.example", "kafka-1",
         "vault-east", "files.cdn.example", "auth.example.net", "10.0.4.17", "10.2.8.40")
TABLES = ("orders", "invoices", "sessions", "carts", "users", "shipments")


def _n(r, lo, hi):
    return r.randint(lo, hi)


def _hex(r, k=6):
    return "".join(r.choice("0123456789abcdef") for _ in range(k))


# Each event: the ERROR message, a lexical question, a paraphrase question.
# The lexical question's rare word and the paraphrase's cleanliness are both
# re-checked per generated log (`common.rare_shared`, `common.paraphrase_clean`).
EVENTS = {
    "checksum": (lambda r: f"checksum mismatch on block {_n(r, 1000, 99999)}, replica {_n(r, 1, 9)} quarantined",
                 "Which line reports a checksum mismatch?",
                 "Which line says stored data failed its integrity verification?"),
    "certificate": (lambda r: f"TLS handshake with {r.choice(HOSTS)} failed: certificate expired",
                    "Which line reports an expired certificate?",
                    "Which line says a secure connection could not be set up because a credential had lapsed?"),
    "oom": (lambda r: f"out of memory, process {_n(r, 100, 9999)} killed by the kernel",
            "Which line reports running out of memory?",
            "Which line says something was terminated for exhausting RAM?"),
    "deadlock": (lambda r: f"deadlock detected on table {r.choice(TABLES)}, transaction {_n(r, 10000, 99999)} rolled back",
                 "Which line reports a deadlock?",
                 "Which line says two database operations blocked each other so one was undone?"),
    "provider": (lambda r: f"payment provider returned 503 for charge ch_{_hex(r)}",
                 "Which line reports a 503 from the payment provider?",
                 "Which line says the card processing company was unavailable?"),
    "ratelimit": (lambda r: f"rate limit exceeded for key k_{_hex(r)}, requests throttled",
                  "Which line says a rate limit was exceeded?",
                  "Which line says a client was slowed down for sending too many calls?"),
    "dns": (lambda r: f"DNS lookup for {r.choice(HOSTS)} timed out",
            "Which line reports a DNS lookup timing out?",
            "Which line says name resolution for a server never got an answer?"),
    "migration": (lambda r: f"schema migration {_n(r, 100, 999)} failed: column already exists",
                  "Which line reports a failed schema migration?",
                  "Which line says an upgrade to the database layout could not be applied?"),
    "queue": (lambda r: f"message queue q_{_hex(r, 4)} is full, dropping events",
              "Which line says a message queue is full?",
              "Which line says a buffer between services overflowed and lost data?"),
    "smtp": (lambda r: f"SMTP relay refused mail to user{_n(r, 10, 999)}@example.org",
             "Which line reports the SMTP relay refusing mail?",
             "Which line says an email could not be delivered because the outgoing server declined it?"),
    "config": (lambda r: f"config key feature.{_hex(r, 4)} missing, falling back to defaults",
               "Which line reports a missing config key?",
               "Which line says a setting was absent so standard values were used instead?"),
    "lock": (lambda r: f"lock on job j{_n(r, 100, 999)} held for {_n(r, 300, 900)}s, forcing release",
             "Which line reports a lock being forced to release?",
             "Which line says an exclusive claim on a task lasted too long and was broken?"),
    "upload": (lambda r: f"S3 upload of export-{_n(r, 1000, 9999)}.csv failed: access denied",
               "Which line reports access denied on an S3 upload?",
               "Which line says a file could not be sent to cloud storage for lack of permission?"),
    "clock": (lambda r: f"clock skew of {_n(r, 800, 5000)}ms against NTP server",
              "Which line reports clock skew?",
              "Which line says the machine's time drifted away from the reference source?"),
    "segfault": (lambda r: f"segfault in libimage.so while resizing img_{_hex(r)}.png",
                 "Which line reports a segfault?",
                 "Which line says a native library crashed during picture processing?"),
    "lag": (lambda r: f"replica lag reached {_n(r, 30, 600)}s, reads redirected to primary",
            "Which line reports replica lag?",
            "Which line says a copy of the database fell behind the leader?"),
    "jwt": (lambda r: f"JWT signature invalid for session s_{_hex(r)}",
            "Which line reports an invalid JWT signature?",
            "Which line says a login token failed its authenticity test?"),
    "breaker": (lambda r: f"circuit breaker opened for {r.choice(SERVICES)} after {_n(r, 5, 20)} consecutive failures",
                "Which line says a circuit breaker opened?",
                "Which line says calls to a dependency were cut off after repeated faults?"),
    "utf8": (lambda r: f"invalid UTF-8 in field name of row {_n(r, 1000, 99999)}",
             "Which line reports invalid UTF-8?",
             "Which line says some text had a broken character encoding?"),
    "cron": (lambda r: f"cron job nightly-{_hex(r, 4)} exited with status 137",
             "Which line reports a cron job exiting with status 137?",
             "Which line says a periodic task ended abnormally?"),
    "index": (lambda r: f"index idx_{r.choice(TABLES)} corrupted, rebuilding from snapshot",
              "Which line reports a corrupted index?",
              "Which line says a lookup structure was damaged and is being recreated?"),
    "websocket": (lambda r: f"websocket closed unexpectedly by client {r.choice(HOSTS[-2:])}",
                  "Which line reports a websocket closing unexpectedly?",
                  "Which line says a live browser link dropped without warning?"),
    "pool": (lambda r: f"thread pool exhausted, {_n(r, 20, 900)} tasks rejected",
             "Which line says a thread pool was exhausted?",
             "Which line says there were no free workers left to take on new work?"),
    "license": (lambda r: f"license for module {_hex(r, 4)} expired yesterday",
                "Which line reports an expired license?",
                "Which line says a paid entitlement is no longer valid?"),
    "backup": (lambda r: f"backup to vault {_n(r, 1, 9)} incomplete: {_n(r, 2, 5)} of 12 chunks missing",
               "Which line reports an incomplete backup?",
               "Which line says a safety copy did not finish?"),
    "ecc": (lambda r: f"GPU {_n(r, 0, 7)} fell off the bus, {_n(r, 2, 64)} ECC errors",
            "Which line reports ECC errors?",
            "Which line says a graphics card disappeared from the machine?"),
    "overflow": (lambda r: f"stack overflow in parser at depth {_n(r, 5000, 90000)}",
                 "Which line reports a stack overflow?",
                 "Which line says recursion went too deep while reading input?"),
    "webhook": (lambda r: f"webhook to hooks.example.com/{_hex(r, 4)} returned 410, subscription disabled",
                "Which line reports a webhook returning 410?",
                "Which line says a callback endpoint no longer exists, so notifications were switched off?"),
    "shard": (lambda r: f"shard {_n(r, 1, 64)} unreachable, query returned partial results",
              "Which line reports an unreachable shard?",
              "Which line says one slice of the dataset could not be contacted, so an answer was incomplete?"),
    "stock": (lambda r: f"negative stock for SKU {_n(r, 10000, 99999)} after order {_n(r, 100000, 999999)}",
              "Which line reports negative stock?",
              "Which line says inventory counts dropped below zero?"),
}

# Routine traffic: never a target, never a distractor.
ROUTINE = {
    "INFO": (
        lambda r: f"GET /v1/{r.choice(TABLES)}/{_n(r, 1, 999)} 200 in {_n(r, 2, 99)}ms",
        lambda r: f"POST /v1/{r.choice(TABLES)} 201 in {_n(r, 5, 99)}ms",
        lambda r: f"user {_n(r, 10, 999)} signed in",
        lambda r: f"cache hit rate {_n(r, 70, 99)}%",
        lambda r: f"flushed {_n(r, 10, 99)} metrics to collector",
        lambda r: f"processed batch {_n(r, 10, 999)} with {_n(r, 10, 99)} rows",
        lambda r: "health probe ok",
        lambda r: f"settings reloaded, {_n(r, 10, 90)} keys",
        lambda r: f"session s_{_hex(r, 4)} renewed",
        lambda r: f"connected to peer {r.choice(HOSTS)}",
        lambda r: f"task t{_n(r, 100, 999)} finished in {_n(r, 1, 90)}s",
        lambda r: f"sent {_n(r, 1, 300)} notifications",
    ),
    "DEBUG": (
        lambda r: f"pool size {_n(r, 4, 64)}, idle {_n(r, 0, 16)}",
        lambda r: f"gc pause {_n(r, 1, 40)}ms",
        lambda r: f"loaded {_n(r, 10, 99)} feature flags",
        lambda r: f"retrying fetch of {r.choice(TABLES)} (attempt {_n(r, 2, 4)})",
        lambda r: f"heartbeat from node {_n(r, 1, 32)}",
        lambda r: f"span {_hex(r, 4)} closed",
    ),
    "WARN": (
        lambda r: f"slow query {_n(r, 500, 999)}ms on {r.choice(TABLES)}",
        lambda r: f"retry {_n(r, 1, 3)} for request r_{_hex(r, 4)}",
        lambda r: f"queue depth {_n(r, 100, 999)} above soft limit",
    ),
}
ROUTINE_WEIGHTS = (("INFO", 0.62), ("DEBUG", 0.3), ("WARN", 0.08))


def _level(r):
    x = r.random()
    for level, weight in ROUTINE_WEIGHTS:
        if x < weight:
            return level
        x -= weight
    return ROUTINE_WEIGHTS[-1][0]


def _routine(r):
    level = _level(r)
    return level, r.choice(ROUTINE[level])(r)


def _stamp(seconds):
    # Minutes, not seconds: every digit is a token here, and a 1,000-line log
    # is meant to be about 16K tokens (spec 18), not 21K.
    seconds %= 86400
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}"


def one_log(r, length, event, split, absent):
    """One question: the log's lines, the target index, the distractors.
    `None` when this draw failed its word checks (the caller redraws)."""
    others = [key for key in EVENTS if key != event]
    distractors = r.sample(others, r.randint(3, 8))
    slots = r.sample(range(length), 1 + len(distractors))
    target_at, distractor_at = slots[0], slots[1:]
    special = {target_at: event, **dict(zip(distractor_at, distractors))}
    clock = r.randint(0, 86399)
    lines = []
    for i in range(length):
        clock += r.randint(0, 20)
        if i in special:
            level, message = "ERROR", EVENTS[special[i]][0](r)
        else:
            level, message = _routine(r)
        service = r.choice(SERVICES)
        lines.append(f"{_stamp(clock)} {level} {service}: {message}")
    question = EVENTS[event][1 if split == "lexical" else 2]
    target = lines[target_at]
    rest = lines[:target_at] + lines[target_at + 1:]
    if split == "lexical" and not rare_shared(question, target, rest):
        return None
    if split == "paraphrase" and not paraphrase_clean(question, target):
        return None
    if absent:
        level, message = _routine(r)
        lines[target_at] = f"{target[:5]} {level} {r.choice(SERVICES)}: {message}"
    return {
        "state": "\n".join(lines),
        "instruction": question,
        "targets": [] if absent else [target_at],
        "distractors": sorted(distractor_at),
        "event": event,
    }


def generate(r, count, prefix="logs"):
    absent = assign_absent(r, count)
    rows = []
    for i in range(count):
        length = LENGTHS[i % len(LENGTHS)]
        split = ("lexical", "paraphrase")[i % 2]
        event = r.choice(sorted(EVENTS))
        for _ in range(50):
            row = one_log(r, length, event, split, i in absent)
            if row is not None:
                break
        else:
            raise SystemExit(f"logs: event {event!r} ({split}) failed its word checks 50 times")
        rows.append({"id": f"{prefix}-{i:03}", "family": "logs", "split": split,
                     "absent": i in absent, "segments": length, **row})
    return rows
