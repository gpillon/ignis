"""pytest: the tool's modules are flat files beside the tests, and no test touches the GPU.

CUDA is hidden from this process before torch is imported (ADR 0006). `CUDA_VISIBLE_DEVICES` is `-1`, not
empty: Windows drops an empty variable.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def no_gpu():
    yield
    assert not torch.cuda.is_initialized(), "a test initialized CUDA"
