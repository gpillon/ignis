"""A BF16 checkpoint with another one's changed tensors laid over it (spec #386: the abliterated BF16
"may be assembled as an overlay on the base rather than a full copy").

The overlay is the directory `replay.py fetch` writes: partial shards holding only the tensors the
variant changed, under the base shards' file names, their index and `fetch.json` (the repository,
the revision and each tensor's sha256). `Reference(root, overlay)` says which file holds each tensor
of the checkpoint so assembled: the overlay's for the tensors it holds, the base's for every other.
A harness that reads the reference through it (GitHub #422's KLD reference) never copies the base.

The overlay is checked before any tensor is read, ValueError naming what is wrong: `fetch.json`
must name exactly the tensors its index maps, in the same shard; each must be one of the base's,
with the base's dtype and shape, so the overlay swaps like for like; and each tensor's bytes must
hash to `fetch.json`'s sha256.
"""
import os

import exl3
from override import FETCH_RECORD, INDEX, Fetched, read_json


class Reference:
    """The BF16 checkpoint at `root`, with `overlay` (a `replay.py fetch` directory) laid over it
    when given. `files` maps every tensor name to the file holding it; `replaced` lists the tensors
    the overlay holds; `repo` and `revision` are the overlay's source (None without one)."""

    def __init__(self, root, overlay=None):
        weight_map = read_json(os.path.join(root, INDEX))["weight_map"]
        self.files = {name: os.path.join(root, shard) for name, shard in weight_map.items()}
        self.replaced, self.repo, self.revision = [], None, None
        if overlay is None:
            return
        fetched = Fetched(overlay)
        stated, mapped = fetched.tensors, fetched.weight_map
        problems = [f"{name}: the index maps it, {FETCH_RECORD} does not name it"
                    for name in sorted(set(mapped) - set(stated))]
        problems += [f"{name}: {FETCH_RECORD} names it, the index does not map it"
                     for name in sorted(set(stated) - set(mapped))]
        headers = {}

        def header(file):
            if file not in headers:
                headers[file] = exl3.read_header(file)
            return headers[file]

        for name in sorted(set(mapped) & set(stated)):
            misplaced = fetched.misplaced(name)
            if misplaced:
                problems.append(misplaced)
                continue
            if name not in self.files:
                problems.append(f"{name}: the base has no tensor of that name")
                continue
            mine, base = header(os.path.join(overlay, mapped[name]))[name], header(self.files[name])[name]
            if (mine.dtype, mine.shape) != (base.dtype, base.shape):
                problems.append(f"{name}: {mine.dtype} {mine.shape}, the base's is {base.dtype} {base.shape}")
                continue
            _, tampered = fetched.digest(name)
            if tampered:
                problems.append(tampered)
        if problems:
            raise ValueError(f"the overlay {overlay} cannot be laid over {root}:\n  " + "\n  ".join(problems))
        for name in mapped:
            self.files[name] = os.path.join(overlay, mapped[name])
        self.replaced = sorted(mapped)
        self.repo, self.revision = fetched.record.get("repo"), fetched.record.get("revision")
