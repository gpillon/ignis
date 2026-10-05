"""The command's guards that need no GPU: one work tree, one configuration."""
import os

import convert


class Stub:
    def __init__(self, work, **run):
        self.work = work
        self.run = run

    def run_record(self):
        return dict(self.run)


def test_a_work_tree_of_another_configuration_is_refused(tmp_path):
    os.makedirs(tmp_path / "state")
    first = Stub(str(tmp_path), layers=48, table_shards=128, budget=2.5)
    assert convert.check_run_record(first) is None                       # recorded
    assert convert.check_run_record(Stub(str(tmp_path), layers=48, table_shards=128, budget=2.5)) is None
    why = convert.check_run_record(Stub(str(tmp_path), layers=2, table_shards=4, budget=2.5))
    assert "layers" in why and "table_shards" in why and "budget" not in why
    assert "budget" in convert.check_run_record(Stub(str(tmp_path), layers=48, table_shards=128, budget=3.0))


def test_the_disk_estimate_is_the_readme_figure():
    assert 73e9 < convert.expected_output_bytes(48, 128) < 75e9
