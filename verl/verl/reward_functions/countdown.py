from __future__ import annotations

from typing import Any

from verl.utils.reward_score.countdown import compute_score as _compute_score


def compute_countdown_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Adapter for Countdown reward.

    Compatible with `custom_reward_function` which passes `(data_source, solution_str, ground_truth, extra_info, ...)`.
    """
    _ = (data_source,)  # unused but part of the expected signature
    return _compute_score(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)


__all__ = ["compute_countdown_score"]
