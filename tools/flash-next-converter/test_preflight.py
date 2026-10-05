"""The busy-GPU refusal (spec 01 acceptance 1)."""
import pytest

import preflight


def test_known_gpu_workloads_are_named_and_desktop_apps_are_not():
    names = ["chrome.exe", "ignis-server.exe", "ninfer-serve.exe", "core-a1b2_gpu-3f.exe", "dwm.exe", "python.exe"]
    assert preflight.gpu_holders(names) == ["core-a1b2_gpu-3f.exe", "ignis-server.exe", "ninfer-serve.exe"]


def test_a_lock_held_by_someone_else_refuses(tmp_path):
    (tmp_path / "owner").write_text("owner=kern\npurpose=x\n")
    with pytest.raises(preflight.Refused, match="kern"):
        preflight.check_gpu("flash-next-convert", lock_dir=str(tmp_path))
    with pytest.raises(preflight.Refused, match="None"):
        preflight.check_gpu("flash-next-convert", lock_dir=str(tmp_path / "absent"))


def test_disk_refuses_below_need_plus_margin(tmp_path):
    with pytest.raises(preflight.Refused, match="margin"):
        preflight.check_disk(str(tmp_path), 10 ** 15, 15 * 10 ** 9, "output")
    assert preflight.check_disk(str(tmp_path), 1, 0, "output") > 0
