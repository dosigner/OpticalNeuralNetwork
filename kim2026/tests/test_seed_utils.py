from __future__ import annotations

import random

import numpy as np
import torch

from kim2026.utils.seed import (
    capture_rng_state,
    restore_rng_state,
    set_global_seed,
)


def test_set_global_seed_makes_rng_reproducible() -> None:
    set_global_seed(123, strict_reproducibility=True)
    py_first = random.random()
    np_first = np.random.rand(3)
    torch_first = torch.rand(3)

    set_global_seed(123, strict_reproducibility=True)
    py_second = random.random()
    np_second = np.random.rand(3)
    torch_second = torch.rand(3)

    assert py_first == py_second
    assert np.array_equal(np_first, np_second)
    assert torch.equal(torch_first, torch_second)


def test_capture_and_restore_rng_state_restores_streams() -> None:
    set_global_seed(321, strict_reproducibility=True)
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1)

    state = capture_rng_state()
    expected = (random.random(), np.random.rand(2), torch.rand(2))

    restore_rng_state(state)
    actual = (random.random(), np.random.rand(2), torch.rand(2))

    assert expected[0] == actual[0]
    assert np.array_equal(expected[1], actual[1])
    assert torch.equal(expected[2], actual[2])
