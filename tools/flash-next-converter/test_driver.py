"""The layer loop: resume per layer, crash-safe checkpoint, stop file (spec 01 acceptance 1).

A toy model stands in for the conversion: its state is one tensor, layer L doubles it
and adds L, and a finished layer leaves a work directory with DONE. The real pipeline
plugs into the same loop.
"""
import os

import pytest
import torch

import driver
import layout

N = 6


class Toy:
    def __init__(self, work, crash_at=None):
        self.work = work
        self.crash_at = crash_at
        self.processed, self.replayed = [], []

    def fingerprint(self):
        return "toy-v1"

    def initial_state(self):
        return {"x": torch.arange(4, dtype=torch.float64)}

    def layer_dir(self, L):
        return os.path.join(self.work, f"L{L:02d}")

    def process(self, L, state):
        d = self.layer_dir(L)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "w.bin"), "wb") as f:
            f.write(b"partial")
        if L == self.crash_at:
            raise RuntimeError("crash")
        state["x"] = state["x"] * 2 + L
        with open(os.path.join(d, "w.bin"), "wb") as f:
            f.write(bytes([L]) * 8)
        layout.mark_done(d)
        self.processed.append(L)

    def replay(self, L, state):
        assert layout.is_done(self.layer_dir(L))
        state["x"] = state["x"] * 2 + L
        self.replayed.append(L)


def expected():
    x = torch.arange(4, dtype=torch.float64)
    for L in range(N):
        x = x * 2 + L
    return x


def loop(tmp_path, toy, stop=None, every=2):
    ck = driver.Checkpoint([str(tmp_path / "ckA"), str(tmp_path / "ckB")], placement=lambda name: 0)
    return driver.LayerLoop(N, toy.layer_dir, stop_file=stop, checkpoint=ck, every=every)


def test_a_full_run_converts_every_layer_once(tmp_path):
    toy = Toy(str(tmp_path / "work"))
    code, state = loop(tmp_path, toy).run(toy)
    assert code == driver.EXIT_DONE
    assert torch.equal(state["x"], expected())
    assert toy.processed == list(range(N)) and toy.replayed == []
    assert all(layout.is_done(toy.layer_dir(L)) for L in range(N))


def test_the_stop_file_exits_75_after_the_layer_and_the_relaunch_resumes(tmp_path):
    stop = tmp_path / "STOP"
    toy = Toy(str(tmp_path / "work"))
    lp = loop(tmp_path, toy, stop=str(stop), every=100)
    orig = toy.process

    def process(L, state):
        orig(L, state)
        if L == 2:
            stop.write_text("")
    toy.process = process
    code, _ = lp.run(toy)
    assert code == driver.EXIT_STOPPED == 75
    assert toy.processed == [0, 1, 2]
    stop.unlink()
    toy2 = Toy(str(tmp_path / "work"))
    code, state = loop(tmp_path, toy2, stop=str(stop), every=100).run(toy2)
    assert code == driver.EXIT_DONE
    assert toy2.processed == [3, 4, 5] and toy2.replayed == []   # the stop wrote a checkpoint
    assert torch.equal(state["x"], expected())


def test_a_stop_file_present_at_launch_stops_before_any_work(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("")
    toy = Toy(str(tmp_path / "work"))
    code, _ = loop(tmp_path, toy, stop=str(stop)).run(toy)
    assert code == driver.EXIT_STOPPED and toy.processed == []


def test_a_crash_keeps_finished_layers_and_redoes_only_the_torn_one(tmp_path):
    toy = Toy(str(tmp_path / "work"), crash_at=3)
    with pytest.raises(RuntimeError):
        loop(tmp_path, toy, every=2).run(toy)
    assert not layout.is_done(toy.layer_dir(3))
    toy2 = Toy(str(tmp_path / "work"))
    code, state = loop(tmp_path, toy2, every=2).run(toy2)
    assert code == driver.EXIT_DONE
    # checkpoint at 2 (after layers 0-1); layer 2 is finished, so it replays; 3 is redone
    assert toy2.replayed == [2] and toy2.processed == [3, 4, 5]
    assert torch.equal(state["x"], expected())
    assert open(os.path.join(toy2.layer_dir(3), "w.bin"), "rb").read() == bytes([3]) * 8


def test_a_torn_checkpoint_is_never_loaded_and_the_finished_layers_replay(tmp_path):
    toy = Toy(str(tmp_path / "work"), crash_at=4)
    lp = loop(tmp_path, toy, every=2)
    with pytest.raises(RuntimeError):
        lp.run(toy)
    lp.checkpoint.invalidate()          # what a crash in the middle of a checkpoint write leaves
    toy2 = Toy(str(tmp_path / "work"))
    code, state = loop(tmp_path, toy2, every=2).run(toy2)
    assert code == driver.EXIT_DONE
    assert toy2.replayed == [0, 1, 2, 3] and toy2.processed == [4, 5]
    assert torch.equal(state["x"], expected())


def test_a_checkpoint_of_another_configuration_is_ignored(tmp_path):
    toy = Toy(str(tmp_path / "work"))
    loop(tmp_path, toy, stop=None, every=2).run(toy)
    toy2 = Toy(str(tmp_path / "work"))
    toy2.fingerprint = lambda: "toy-v2"
    code, state = loop(tmp_path, toy2, every=2).run(toy2)
    assert toy2.replayed == list(range(N)) and toy2.processed == []
    assert torch.equal(state["x"], expected())


def test_checkpoint_files_are_spread_over_their_directories(tmp_path):
    ck = driver.Checkpoint([str(tmp_path / "big"), str(tmp_path / "small")],
                           placement=lambda name: 0 if name.startswith("bf16") else 1)
    ck.save({"bf16.cal": torch.ones(3), "q.test": torch.zeros(2), "stats": {"a": 1}}, 4, "fp")
    assert sorted(os.listdir(tmp_path / "big")) == ["bf16.cal.pt", "meta.json"]
    assert "q.test.pt" in os.listdir(tmp_path / "small")
    state, nxt = ck.load("fp")
    assert nxt == 4 and torch.equal(state["bf16.cal"], torch.ones(3)) and state["stats"] == {"a": 1}
