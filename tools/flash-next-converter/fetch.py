"""Checkpoint tensors fetched straight into RAM with HTTP range requests, at a pinned revision.

Ported from the compression study's fetcher (`real/fetch.py`): each safetensors header
is read once (cached as JSON in the work tree), then a tensor's byte range is pulled with
parallel range requests on the CDN URL the hub redirects to. Nothing touches a disk but
the small caches. The study fetched `main`; this fetches the revision the converter is
pinned to, and the index comes from that revision too.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import torch

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PART = 32 << 20
WORKERS = 16
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64,
      "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}


class Source:
    def __init__(self, cache_dir, repo=REPO, revision=REVISION, log=print):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.cache_dir = cache_dir
        self.log = log
        self._local = threading.local()
        self._urls = {}
        self._url_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(WORKERS)
        os.makedirs(cache_dir, exist_ok=True)
        self.index = self._json_cached("model.safetensors.index.json")["weight_map"]
        self.headers = self._headers()

    # -------------------------------------------------------------- HTTP
    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = httpx.Client(timeout=120, follow_redirects=True)
        return s

    def _cdn(self, f, refresh=False):
        with self._url_lock:
            if f in self._urls and not refresh:
                return self._urls[f]
        r = httpx.head(self.base + f, follow_redirects=False, timeout=60)
        loc = r.headers.get("Location") or r.headers.get("location")
        url = loc if loc else self.base + f
        if url.startswith("/"):
            url = "https://huggingface.co" + url
        with self._url_lock:
            self._urls[f] = url
        return url

    def _range(self, f, start, end):
        """Bytes [start, end) of file f, with retries and a URL refresh."""
        last = None
        for attempt in range(10):
            try:
                r = self._session().get(self._cdn(f, refresh=attempt > 0),
                                        headers={"Range": f"bytes={start}-{end - 1}"})
                if r.status_code in (200, 206) and len(r.content) == end - start:
                    return r.content
                raise IOError(f"status {r.status_code} len {len(r.content)}")
            except Exception as e:  # network errors of any kind: retry with backoff
                last = e
                time.sleep(2 + 3 * attempt)
        raise IOError(f"range {f} {start}-{end}: {last!r}")

    def file_bytes(self, f):
        """A whole small file of the revision (config, tokenizer, index)."""
        for attempt in range(10):
            try:
                r = self._session().get(self.base + f)
                if r.status_code == 200:
                    return r.content
                raise IOError(f"{f}: status {r.status_code}")
            except Exception as e:
                last = e
                time.sleep(2 + 3 * attempt)
        raise IOError(f"{f}: {last!r}")

    def _json_cached(self, f):
        path = os.path.join(self.cache_dir, f)
        if not os.path.exists(path):
            raw = self.file_bytes(f)
            with open(path + ".tmp", "wb") as out:
                out.write(raw)
            os.replace(path + ".tmp", path)
        return json.load(open(path))

    def _headers(self):
        path = os.path.join(self.cache_dir, "headers.json")
        if os.path.exists(path):
            return json.load(open(path))

        def one(f):
            n = int.from_bytes(self._range(f, 0, 8), "little")
            h = json.loads(self._range(f, 8, 8 + n))
            h.pop("__metadata__", None)
            return f, {"base": 8 + n, "t": h}
        out = dict(self._pool.map(one, sorted(set(self.index.values()))))
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        return out

    # -------------------------------------------------------------- tensors
    def meta(self, name):
        return self.headers[self.index[name]]["t"][name]

    def size(self, name):
        a, b = self.meta(name)["data_offsets"]
        return b - a

    def get(self, name):
        """One tensor, fetched with parallel range requests; it keeps its bytearray (no copy)."""
        f = self.index[name]
        meta = self.meta(name)
        s0, s1 = meta["data_offsets"]
        base = self.headers[f]["base"]
        buf = bytearray(s1 - s0)
        parts = [(a, min(a + PART, s1)) for a in range(s0, s1, PART)]

        def one(p):
            a, b = p
            buf[a - s0:b - s0] = self._range(f, base + a, base + b)
        list(self._pool.map(one, parts))
        if not len(buf):
            return torch.empty(meta["shape"], dtype=DT[meta["dtype"]])
        return torch.frombuffer(buf, dtype=DT[meta["dtype"]]).reshape(meta["shape"])

    def names(self, prefix, skip=()):
        return [k for k in self.index if k.startswith(prefix) and not k[len(prefix):].startswith(skip)]

    def load(self, prefix, skip=()):
        """{suffix: tensor} for every tensor under prefix; big tensors one after another
        (each already parallel), small ones after."""
        out = {}
        for k in sorted(self.names(prefix, skip), key=lambda k: -self.size(k)):
            out[k[len(prefix):]] = self.get(k)
        return out


class Prefetch:
    """Loads layer L+1's tensors on a thread while layer L computes (depth 1 or 0)."""

    def __init__(self, load_layer, depth=1):
        self.load_layer = load_layer
        self.depth = depth
        self.res, self.th, self.err = {}, {}, {}

    def start(self, L):
        if L in self.th:
            return

        def run():
            try:
                self.res[L] = self.load_layer(L)
            except BaseException as e:
                self.err[L] = e
        self.th[L] = threading.Thread(target=run, daemon=True)
        self.th[L].start()

    def get(self, L, n_layers):
        self.start(L)
        self.th.pop(L).join()
        if L in self.err:
            raise self.err.pop(L)
        out = self.res.pop(L)
        if self.depth and L + 1 < n_layers:
            self.start(L + 1)
        return out
