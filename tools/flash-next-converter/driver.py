"""The layer loop: one decoder layer at a time, resumable, stoppable, crash-safe.

A layer is *finished* when its work directory holds DONE (layout.md §2) and its files still
match it (a torn directory is wiped). On launch the loop loads the state checkpoint if it
is complete, of this configuration and not past an unfinished layer, then
walks the layers: a finished layer is **replayed** (the streams run forward from the
work files, nothing is re-encoded), an unfinished one is wiped and **processed**.
Without a usable checkpoint the replay starts at layer 0, so a torn checkpoint costs
time, never the finished layers.

The checkpoint is a single slot spread over one or more directories (the BF16 stream on
one drive, the smaller streams on another): the disks here hold one copy, not two. It
is written every `every` layers, after the last layer, and on every clean stop. Before a
write it checks that each drive has room for the new slot (counting the old slot's files,
which go first); without room it leaves the old slot untouched and raises NoRoom. It is
marked invalid before its old files are removed and valid only after every new file is
written, so a crash mid-write leaves an invalid slot, never a mixed one.

Before each layer the loop checks the stop file (exit EXIT_STOPPED, 75) and the space the
layer will write (`space_check`; exit EXIT_DISK, 4): either way it checkpoints first, so
the relaunch resumes where it stopped.
"""
import json
import os
import shutil
import time

import torch

import layout

EXIT_DONE, EXIT_ERROR, EXIT_REFUSED, EXIT_QUALITY_FAIL, EXIT_DISK, EXIT_STOPPED = 0, 1, 2, 3, 4, 75
GB = 1e9


class NoRoom(RuntimeError):
    pass


def _free(path):
    import preflight
    return preflight.free_bytes(path)


class Checkpoint:
    def __init__(self, dirs, placement, free=_free, headroom=1 * GB):
        """`dirs`: slot directories; `placement(name) -> index into dirs` for each state entry.
        Entries that are not tensors go into the first directory's meta.json. `free(dir)` gives
        a drive's free bytes; a write keeps `headroom` free on every drive it touches."""
        self.dirs = list(dirs)
        self.placement = placement
        self.free = free
        self.headroom = headroom

    def _meta_path(self):
        return os.path.join(self.dirs[0], "meta.json")

    def invalidate(self):
        os.makedirs(self.dirs[0], exist_ok=True)
        layout.write_json_atomic(self._meta_path(), {"status": "writing"})

    def _check_room(self, state):
        need = {}
        for name, v in state.items():
            if torch.is_tensor(v):
                d = self.dirs[self.placement(name)]
                need[d] = need.get(d, 0) + v.numel() * v.element_size()
        for d, n in need.items():
            os.makedirs(d, exist_ok=True)
            old = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                      if f.endswith(".pt") or f.endswith(".pt.tmp"))
            if self.free(d) + old < n + self.headroom:
                raise NoRoom(f"checkpoint: {d} has {(self.free(d) + old) / GB:.1f} GB for a "
                             f"{n / GB:.1f} GB slot (+{self.headroom / GB:.0f} GB headroom)")

    def save(self, state, next_layer, fingerprint, log=print):
        t0 = time.time()
        self._check_room(state)
        self.invalidate()
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)
            for name in os.listdir(d):
                if name.endswith(".pt") or name.endswith(".pt.tmp"):
                    os.remove(os.path.join(d, name))
        files, small, nbytes = {}, {}, 0
        for name, v in state.items():
            if not torch.is_tensor(v):
                small[name] = v
                continue
            d = self.dirs[self.placement(name)]
            path = os.path.join(d, name + ".pt")
            torch.save(v, path + ".tmp")          # the tensor itself, never a clone
            os.replace(path + ".tmp", path)
            files[name] = path
            nbytes += v.numel() * v.element_size()
        layout.write_json_atomic(self._meta_path(), {"status": "complete", "next_layer": next_layer,
                                                     "fingerprint": fingerprint, "files": files,
                                                     "small": small})
        log(f"checkpoint: next layer {next_layer}, {nbytes / 1e9:.1f} GB in {time.time() - t0:.0f}s")

    def _meta(self, fingerprint):
        try:
            meta = json.load(open(self._meta_path()))
        except (OSError, ValueError):
            return None
        if meta.get("status") != "complete" or meta.get("fingerprint") != fingerprint:
            return None
        return meta

    def next_layer(self, fingerprint):
        """The layer a complete checkpoint of this run resumes at, without loading it."""
        meta = self._meta(fingerprint)
        return meta["next_layer"] if meta else None

    def load(self, fingerprint):
        """(state, next_layer), or None when there is no complete checkpoint of this run."""
        meta = self._meta(fingerprint)
        if meta is None:
            return None
        state = dict(meta["small"])
        for name, path in meta["files"].items():
            state[name] = torch.load(path, weights_only=True)
        return state, meta["next_layer"]


class LayerLoop:
    def __init__(self, n_layers, layer_dir, stop_file=None, checkpoint=None, every=0, log=print,
                 space_check=None):
        """`space_check(L)` returns None when layer L may write, or why the disk is too full."""
        self.n = n_layers
        self.layer_dir = layer_dir
        self.stop_file = stop_file
        self.checkpoint = checkpoint
        self.every = every
        self.log = log
        self.space_check = space_check

    def _stop_requested(self):
        return bool(self.stop_file) and os.path.exists(self.stop_file)

    def _save(self, state, next_layer, fp):
        """Checkpoints; False (old slot untouched) when a drive lacks the room."""
        try:
            self.checkpoint.save(state, next_layer, fp, self.log)
            return True
        except NoRoom as e:
            self.log(f"{e}: the previous checkpoint is kept")
            return False

    def run(self, model):
        """model: fingerprint(), initial_state(), process(L, state), replay(L, state).
        Returns (exit code, state)."""
        fp = model.fingerprint()
        for L in range(self.n):
            d = self.layer_dir(L)
            if layout.is_done(d) and not layout.verify_done(d):
                self.log(f"layer {L}: its files do not match DONE, the layer is redone")
                shutil.rmtree(d)
        first_open = next((L for L in range(self.n) if not layout.is_done(self.layer_dir(L))), self.n)
        got = self.checkpoint.load(fp) if self.checkpoint else None
        if got is not None and got[1] > first_open:
            self.log(f"the checkpoint (layer {got[1]}) is past unfinished layer {first_open}: not used")
            got = None
        if got is not None:
            state, start = got
            self.log(f"resumed from the checkpoint at layer {start}")
        else:
            state, start = model.initial_state(), 0
        dirty = False   # whether the state moved past the checkpoint
        for L in range(start, self.n):
            stop = self._stop_requested()
            full = None if stop or not self.space_check or layout.is_done(self.layer_dir(L)) \
                else self.space_check(L)
            if stop or full:
                if dirty and self.checkpoint:
                    self._save(state, L, fp)
                if stop:
                    self.log(f"stop file {self.stop_file} present: stopping before layer {L}")
                    return EXIT_STOPPED, state
                self.log(f"stopping before layer {L}: {full}")
                return EXIT_DISK, state
            d = self.layer_dir(L)
            if layout.is_done(d):
                model.replay(L, state)
            else:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                model.process(L, state)
            dirty = True
            last = L + 1 == self.n
            if self.checkpoint and (last or (self.every and (L + 1) % self.every == 0)) \
                    and not self._stop_requested():
                if self._save(state, L + 1, fp):
                    dirty = False
                elif not last:
                    return EXIT_DISK, state
        return EXIT_DONE, state
