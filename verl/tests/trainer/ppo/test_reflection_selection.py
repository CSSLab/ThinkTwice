import importlib.util
from pathlib import Path

import numpy as np


def _load_reflection_utils():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl" / "trainer" / "ppo" / "reflection_utils.py"
    spec = importlib.util.spec_from_file_location("reflection_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_wrong_unique_first():
    reflection_utils = _load_reflection_utils()
    uids = np.array(["q1", "q1", "q2", "q2", "q3", "q3"], dtype=object)
    responses = np.array(
        [
            [1, 2, 0],
            [3, 4, 0],
            [5, 6, 0],
            [7, 8, 0],
            [9, 10, 0],
            [11, 12, 0],
        ]
    )
    is_correct = np.array([False, True, False, True, False, True])

    indices = reflection_utils.select_reflection_indices(uids, responses, is_correct, num_select=3, seed=7, step=0)

    assert len(indices) == 3
    assert {uids[i] for i in indices} == {"q1", "q2", "q3"}
    assert all(not is_correct[i] for i in indices)


def test_select_fill_with_correct():
    reflection_utils = _load_reflection_utils()
    uids = np.array(["q1", "q1", "q2", "q2"], dtype=object)
    responses = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    is_correct = np.array([True, True, False, True])

    indices = reflection_utils.select_reflection_indices(uids, responses, is_correct, num_select=3, seed=0, step=0)

    assert len(indices) == 3
    assert any(not is_correct[i] for i in indices)
    assert any(is_correct[i] for i in indices)


def test_select_dedup_responses():
    reflection_utils = _load_reflection_utils()
    uids = np.array(["q1", "q2", "q3"], dtype=object)
    responses = np.array([[1, 2, 0], [1, 2, 0], [3, 4, 0]])
    is_correct = np.array([False, False, True])

    indices = reflection_utils.select_reflection_indices(uids, responses, is_correct, num_select=2, seed=1, step=0)

    assert len(indices) == 2
    response_keys = {responses[i].tobytes() for i in indices}
    assert len(response_keys) == len(indices)


def test_select_respects_response_mask():
    reflection_utils = _load_reflection_utils()
    uids = np.array(["q1", "q2", "q3"], dtype=object)
    responses = np.array([[1, 2, 99], [1, 2, 0], [3, 4, 0]])
    response_mask = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    is_correct = np.array([False, False, True])

    indices = reflection_utils.select_reflection_indices(
        uids, responses, is_correct, num_select=2, response_mask=response_mask, seed=2, step=0
    )

    assert len(indices) == 2
    assert not ({0, 1}.issubset(set(indices)))


def test_select_deterministic_for_seed():
    reflection_utils = _load_reflection_utils()
    uids = np.array(["q1", "q1", "q2", "q2"], dtype=object)
    responses = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    is_correct = np.array([False, False, False, False])

    indices_a = reflection_utils.select_reflection_indices(uids, responses, is_correct, num_select=3, seed=3, step=7)
    indices_b = reflection_utils.select_reflection_indices(uids, responses, is_correct, num_select=3, seed=3, step=7)

    assert indices_a == indices_b
