# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for selecting reflection candidates."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _as_numpy(array: object) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _mix_seed(seed: int, step: int) -> int:
    mixed = (np.uint64(seed) + np.uint64(step) * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return int(mixed)


def _build_response_keys(responses: np.ndarray, response_mask: Optional[np.ndarray]) -> list[tuple[str, tuple[int, ...], bytes]]:
    keys: list[tuple[str, tuple[int, ...], bytes]] = []
    if response_mask is not None:
        response_mask = response_mask.astype(bool)
    for i in range(len(responses)):
        tokens = responses[i]
        if not isinstance(tokens, np.ndarray):
            tokens = np.asarray(tokens)
        if response_mask is not None:
            tokens = tokens[response_mask[i]]
        if tokens.dtype == object:
            tokens = np.asarray(tokens.tolist())
        keys.append((str(tokens.dtype), tokens.shape, tokens.tobytes()))
    return keys


def _select_unique(
    indices: list[int],
    response_keys: list[tuple[str, tuple[int, ...], bytes]],
    rng: np.random.Generator,
    num_select: int,
    selected: list[int],
    selected_set: set[int],
    selected_keys: set[tuple[str, tuple[int, ...], bytes]],
) -> None:
    if not indices or len(selected) >= num_select:
        return
    for idx in rng.permutation(indices).tolist():
        if len(selected) >= num_select:
            break
        key = response_keys[idx]
        if key in selected_keys:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        selected_keys.add(key)


def _select_wrong_first(
    uids: np.ndarray,
    is_correct: np.ndarray,
    response_keys: list[tuple[str, tuple[int, ...], bytes]],
    rng: np.random.Generator,
    num_select: int,
) -> list[int]:
    indices = np.arange(len(is_correct))
    wrong_indices = indices[~is_correct]
    correct_indices = indices[is_correct]

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_keys: set[tuple[str, tuple[int, ...], bytes]] = set()

    uid_to_wrong: dict[object, list[int]] = {}
    for idx in wrong_indices.tolist():
        uid_to_wrong.setdefault(uids[idx], []).append(int(idx))

    uid_list = list(uid_to_wrong.keys())
    if len(uid_list) > 1:
        uid_list = rng.permutation(uid_list).tolist()

    for uid in uid_list:
        if len(selected) >= num_select:
            break
        candidates = uid_to_wrong[uid]
        if len(candidates) > 1:
            candidates = rng.permutation(candidates).tolist()
        for idx in candidates:
            key = response_keys[idx]
            if key in selected_keys:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            selected_keys.add(key)
            break

    if len(selected) < num_select and len(wrong_indices) > 0:
        remaining_wrong = [
            int(idx)
            for idx in wrong_indices.tolist()
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        _select_unique(remaining_wrong, response_keys, rng, num_select, selected, selected_set, selected_keys)

    if len(selected) < num_select and len(correct_indices) > 0:
        remaining_correct = [
            int(idx)
            for idx in correct_indices.tolist()
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        _select_unique(remaining_correct, response_keys, rng, num_select, selected, selected_set, selected_keys)

    return selected


def _select_balanced(
    uids: np.ndarray,
    is_correct: np.ndarray,
    response_keys: list[tuple[str, tuple[int, ...], bytes]],
    rng: np.random.Generator,
    num_select: int,
) -> list[int]:
    indices = np.arange(len(is_correct))
    correct_indices = indices[is_correct]
    wrong_indices = indices[~is_correct]

    uid_to_correct: dict[object, list[int]] = {}
    for idx in correct_indices.tolist():
        uid_to_correct.setdefault(uids[idx], []).append(int(idx))

    uid_to_wrong: dict[object, list[int]] = {}
    for idx in wrong_indices.tolist():
        uid_to_wrong.setdefault(uids[idx], []).append(int(idx))

    uids_with_both = [u for u in uid_to_correct if u in uid_to_wrong]
    if len(uids_with_both) > 1:
        uids_with_both = rng.permutation(uids_with_both).tolist()

    half = num_select // 2
    uids_for_correct = uids_with_both[:half]
    uids_for_wrong = uids_with_both[half : half + (num_select - half)]

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_keys: set[tuple[str, tuple[int, ...], bytes]] = set()

    for uid in uids_for_correct:
        candidates = uid_to_correct[uid]
        if len(candidates) > 1:
            candidates = rng.permutation(candidates).tolist()
        for idx in candidates:
            key = response_keys[idx]
            if key in selected_keys:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            selected_keys.add(key)
            break

    for uid in uids_for_wrong:
        candidates = uid_to_wrong[uid]
        if len(candidates) > 1:
            candidates = rng.permutation(candidates).tolist()
        for idx in candidates:
            key = response_keys[idx]
            if key in selected_keys:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            selected_keys.add(key)
            break

    if len(selected) < num_select:
        remaining = [
            int(idx)
            for idx in range(len(uids))
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        _select_unique(remaining, response_keys, rng, num_select, selected, selected_set, selected_keys)

    return selected


def _select_fair(
    uids: np.ndarray,
    response_keys: list[tuple[str, tuple[int, ...], bytes]],
    rng: np.random.Generator,
    num_select: int,
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_keys: set[tuple[str, tuple[int, ...], bytes]] = set()

    uid_to_indices: dict[object, list[int]] = {}
    for idx, uid in enumerate(uids.tolist()):
        uid_to_indices.setdefault(uid, []).append(int(idx))

    uid_list = list(uid_to_indices.keys())
    if len(uid_list) > 1:
        uid_list = rng.permutation(uid_list).tolist()

    for uid in uid_list:
        if len(selected) >= num_select:
            break
        candidates = uid_to_indices[uid]
        if len(candidates) > 1:
            candidates = rng.permutation(candidates).tolist()
        for idx in candidates:
            key = response_keys[idx]
            if key in selected_keys:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            selected_keys.add(key)
            break

    if len(selected) < num_select:
        remaining = [
            int(idx)
            for idx in range(len(uids))
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        _select_unique(remaining, response_keys, rng, num_select, selected, selected_set, selected_keys)

    return selected


def _select_variance_based(
    uids: np.ndarray,
    is_correct: np.ndarray,
    response_keys: list[tuple[str, tuple[int, ...], bytes]],
    reflection_is_correct: np.ndarray,
    rng: np.random.Generator,
    num_select: int,
    repeat_times: int,
) -> list[int]:
    indices = np.arange(len(is_correct))

    uid_to_indices: dict[object, list[int]] = {}
    for idx in indices:
        uid_to_indices.setdefault(uids[idx], []).append(int(idx))

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_keys: set[tuple[str, tuple[int, ...], bytes]] = set()

    uid_list = list(uid_to_indices.keys())
    if len(uid_list) > 1:
        uid_list = rng.permutation(uid_list).tolist()

    for uid in uid_list:
        if len(selected) >= num_select:
            break

        candidates = uid_to_indices[uid]
        if len(candidates) > 1:
            candidates = rng.permutation(candidates).tolist()

        found = False
        for candidate_idx in candidates:
            if candidate_idx in selected_set:
                continue

            key = response_keys[candidate_idx]
            if key in selected_keys:
                continue

            refl_slice = reflection_is_correct[candidate_idx * repeat_times : (candidate_idx + 1) * repeat_times]
            if len(refl_slice) > 1 and np.std(refl_slice.astype(float)) > 0:
                selected.append(candidate_idx)
                selected_set.add(candidate_idx)
                selected_keys.add(key)
                found = True
                break

        if not found and candidates:
            for candidate_idx in candidates:
                if candidate_idx in selected_set:
                    continue
                key = response_keys[candidate_idx]
                if key not in selected_keys:
                    selected.append(candidate_idx)
                    selected_set.add(candidate_idx)
                    selected_keys.add(key)
                    break

    if len(selected) < num_select:
        remaining = [
            int(idx)
            for idx in range(len(uids))
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        _select_unique(remaining, response_keys, rng, num_select, selected, selected_set, selected_keys)

    return selected


def select_reflection_indices(
    uids: np.ndarray | list,
    responses: np.ndarray | torch.Tensor | list,
    is_correct: np.ndarray | list,
    *,
    num_select: int,
    response_mask: Optional[np.ndarray | torch.Tensor | list] = None,
    seed: int = 0,
    step: int = 0,
    selection_mode: str = "wrong-first",
    reflection_is_correct: Optional[np.ndarray | list] = None,
    repeat_times: int = 1,
) -> list[int]:
    """Select indices for reflection with configurable sampling strategy.

    Modes:
        - wrong-first: prioritize wrong responses
        - fair: uniform across uids, no correctness bias
        - balanced: half correct, half wrong (1 per uid each)
        - variance-based: select base rollouts where reflection shows variance
    Duplicate responses (after applying response_mask) are never selected.
    """

    responses_np = _as_numpy(responses)
    uids_np = _as_numpy(uids).reshape(-1)
    is_correct_np = _as_numpy(is_correct).astype(bool).reshape(-1)
    if response_mask is not None:
        response_mask_np = _as_numpy(response_mask)
    else:
        response_mask_np = None

    total = len(responses_np)
    if num_select <= 0 or total == 0:
        return []
    if len(uids_np) != total or len(is_correct_np) != total:
        raise ValueError("uids, responses, and is_correct must have matching lengths.")

    response_keys = _build_response_keys(responses_np, response_mask_np)
    rng = np.random.default_rng(_mix_seed(int(seed), int(step)))

    mode = str(selection_mode).lower()
    if mode in {"wrong-first", "wrong_first"}:
        return _select_wrong_first(uids_np, is_correct_np, response_keys, rng, num_select)
    if mode in {"uniform"}:
        return _select_fair(uids_np, response_keys, rng, num_select)
    if mode in {"balanced"}:
        return _select_balanced(uids_np, is_correct_np, response_keys, rng, num_select)
    if mode in {"variance-based", "variance_based"}:
        if reflection_is_correct is None:
            raise ValueError("reflection_is_correct required for variance-based mode")
        return _select_variance_based(
            uids_np,
            is_correct_np,
            response_keys,
            _as_numpy(reflection_is_correct).astype(bool),
            rng,
            num_select,
            repeat_times,
        )
    raise ValueError(f"Unknown reflection selection mode: {selection_mode}")
