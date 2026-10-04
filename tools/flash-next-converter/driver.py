"""The layer loop: one decoder layer at a time, resumable, stoppable, crash-safe.

A layer is *finished* when its work directory holds DONE (layout.md §2). On launch the
loop loads the state checkpoint if it is complete and of this configuration, then
walks the layers: a finished layer is **replayed** (the streams run forward from the
work files, nothing is re-encoded), an unfinished one is wiped and **processed**.
Without a usable checkpoint the replay starts at layer 0, so a torn checkpoint costs
time, never the finished layers.

The checkpoint is a single slot spread over one or more directories (the BF16 stream on
one drive, the smaller streams on another): the disks here hold one copy, not two. It
is marked invalid before its old files are removed and valid only after every new file
is written, so a crash mid-write leaves an invalid slot, never a mixed one.

The stop file is checked before each layer: when present the loop checkpoints and
returns EXIT_STOPPED (75), so the coordinator can open a GPU window and relaunch.
"""
import json
import os
import shutil
import time

import torch

import layout

EXIT_DONE, EXIT_ERROR, EXIT_REFUSED, EXIT_QUALITY_FAIL, EXIT_STOPPED = 0, 1, 2, 3, 75


class Checkpoint:
    def __init__(self, dirs, placement):
        """`dirs`: slot directories; `placement(name) -> index into dirs` for each state entry.
        Entries that are not tensors go into the first directory's meta.json."""
        self.dirs = list(dirs)
        self.placement = placement

    def _meta_path(self):
        return os.path.join(self.dirs[0], "meta.json")

    def invalidate(self):
        os.makedirs(self.dirs[0], exist_ok=True)
        layout.write_json_atomic(self._meta_path(), {"status": "writing"})

    def save(self, state, next_layer, fingerprint, log=print):
        t0 = time.time()
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
    def __init__(self, n_layers, layer_dir, stop_file=None, checkpoint=None, every=0, log=print):
        self.n = n_layers
        self.layer_dir = layer_dir
        self.stop_file = stop_file
        self.checkpoint = checkpoint
        self.every = every
        self.log = log

    def _stop_requested(self):
        return bool(self.stop_file) and os.path.exists(self.stop_file)

    def run(self, model):
        """model: fingerprint(), initial_state(), process(L, state), replay(L, state).
        Returns (exit code, state)."""
        fp = model.fingerprint()
        got = self.checkpoint.load(fp) if self.checkpoint else None
        if got is not None:
            state, start = got
            self.log(f"resumed from the checkpoint at layer {start}")
        else:
            state, start = model.initial_state(), 0
        if any(not layout.is_done(self.layer_dir(L)) for L in range(start)):
            raise RuntimeError("the checkpoint is past a layer whose work directory is not finished")
        dirty = False   # whether the state moved past the checkpoint
        for L in range(start, self.n):
            if self._stop_requested():
                if dirty and self.checkpoint:
                    self.checkpoint.save(state, L, fp, self.log)
                self.log(f"stop file {self.stop_file} present: stopping before layer {L}")
                return EXIT_STOPPED, state
            d = self.layer_dir(L)
            if layout.is_done(d):
                model.replay(L, state)
            else:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                model.process(L, state)
            dirty = True
            if self.checkpoint and self.every and (L + 1) % self.every == 0 and L + 1 < self.n \
                    and not self._stop_requested():
                self.checkpoint.save(state, L + 1, fp, self.log)
                dirty = False
        return EXIT_DONE, state
