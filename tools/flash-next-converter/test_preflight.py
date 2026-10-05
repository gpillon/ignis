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


def test_vram_in_use_above_the_guard_threshold_refuses(tmp_path):
    (tmp_path / "owner").write_text("owner=conv\n")
    ok = dict(lock_dir=str(tmp_path), names=lambda: ["dwm.exe"])
    assert preflight.check_gpu("conv", vram_used=lambda: 2900, **ok) == 2900
    with pytest.raises(preflight.Refused, match="VRAM"):
        preflight.check_gpu("conv", vram_used=lambda: preflight.VRAM_THRESHOLD_MIB, **ok)
    with pytest.raises(preflight.Refused, match="ignis-server"):
        preflight.check_gpu("conv", lock_dir=str(tmp_path), names=lambda: ["ignis-server.exe"],
                            vram_used=lambda: 0)


def test_a_missing_drive_is_refused_not_looped_on():
    import string
    import os
    free = next(d for d in reversed(string.ascii_uppercase) if not os.path.exists(f"{d}:/"))
    with pytest.raises(preflight.Refused, match="no such drive"):
        preflight.free_bytes(f"{free}:/flash-next-ckpt")
