"""The calibration and test corpus: the compression study's run 4-8 set, verbatim.

The token files stay outside the repository (study data, ~5 MB); the committed
`corpus_manifest.json` pins each by sha256 and names its sources, and every source must
be in the manifest's allowlist (spec 01: public data or the owner's own, nothing
contributed). The chunk order reproduces the study's `real/e2e8.py` `chunks_all()`:
calibration chunks first, then the held-out ones.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "corpus_manifest.json")
CH = 2048


class CorpusError(ValueError):
    pass


@dataclass
class Corpus:
    chunks: list            # dicts: ids, kind, test, cal, valid, mmlu, source
    letter_ids: list
    long: list              # dicts: ids, kind, source
    manifest: list = field(default_factory=list)

    @property
    def mmlu_questions(self):
        return sum(len(c["mmlu"]) for c in self.chunks if c["test"])


def check_allowlist(man):
    allowed = set(man["allowlist"])
    named = [(name, f["sources"]) for name, f in man["files"].items()] + [("canary", man["canary"]["sources"])]
    for name, sources in named:
        bad = [s for s in sources if s not in allowed]
        if bad:
            raise CorpusError(f"{name}: sources {bad} are not in the corpus allowlist")


def _read(man, name, dirs):
    f = man["files"][name]
    path = os.path.join(dirs[f["dir"]], name)
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != f["sha256"]:
        raise CorpusError(f"{path}: sha256 {got} differs from the manifest's {f['sha256']}")
    return json.loads(raw)


def _window_chunks(windows, metas, long_windows, long_metas):
    """The study's `chunk_list()`: every 4096-token span of the 27B KLD windows, cut in
    two 2048-token chunks; a span is held out when its index is 4 mod 5."""
    pairs = []
    for ws, ms in ((windows, metas), (long_windows, long_metas)):
        for ids, meta in zip(ws, ms):
            for s in range(0, len(ids) - 4096 + 1, 4096):
                pairs.append((ids[s:s + 4096], meta["kind"]))
    out = []
    for p, (ids, kind) in enumerate(pairs):
        for h in range(2):
            out.append({"ids": ids[h * CH:(h + 1) * CH], "kind": kind, "test": p % 5 == 4})
    return out


def load(ood_dir, windows_dir, manifest=None):
    man = manifest or json.load(open(MANIFEST))
    check_allowlist(man)
    dirs = {"ood": str(ood_dir), "windows": str(windows_dir)}
    data = {name: _read(man, name, dirs) for name in man["files"]}
    base = _window_chunks(data["windows.json"], data["windows_meta.json"], data["long_windows.json"],
                          data["long_windows_meta.json"])
    src_w = "windows.json+long_windows.json"
    in_tr = [dict(c, valid=CH, mmlu=[], cal=True, source=src_w) for c in base if not c["test"]][:64]
    in_te = [dict(c, valid=CH, mmlu=[], cal=False, source=src_w) for c in base if c["test"]][:8]
    r4 = [dict(c, source="run4_chunks.json") for c in data["run4_chunks.json"]["chunks"]]
    ood = data["chunks.json"]
    mm_te = [dict(c, cal=False, source="chunks.json") for c in ood["chunks"]
             if c["test"] and c["kind"] == "mmlu"][:30]
    mm_cal = [dict(c, cal=True, source="calib_mmlu.json") for c in data["calib_mmlu.json"]["chunks"]]
    cal = in_tr + [c for c in r4 if c["cal"]] + mm_cal
    tests = in_te + [c for c in r4 if c["test"]] + mm_te
    for c in cal + tests:
        if len(c["ids"]) != CH:
            raise CorpusError(f"a {c['kind']} chunk has {len(c['ids'])} tokens, want {CH}")
    long = []
    for pick in man["long8192"]:
        w = data["long_windows.json"][pick["window"]]
        meta = data["long_windows_meta.json"][pick["window"]]
        if meta["kind"] != pick["kind"]:
            raise CorpusError(f"long window {pick['window']} is {meta['kind']}, the manifest says {pick['kind']}")
        for s in pick["starts"]:
            ids = w[s:s + 8192]
            if len(ids) != 8192:
                raise CorpusError(f"long window {pick['window']} has no 8192 tokens at {s}")
            long.append({"ids": ids, "kind": pick["kind"],
                         "source": f"long_windows.json[{pick['window']}][{s}:{s + 8192}]"})
    entries = [{"file": name, "sha256": f["sha256"], "sources": f["sources"]} for name, f in man["files"].items()]
    return Corpus(cal + tests, ood["letter_ids"], long, entries)
