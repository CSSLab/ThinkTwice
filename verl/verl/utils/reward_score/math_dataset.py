"""
MATH Dataset Reward Function for VERL
Wrapper around the high-recall math grader from understand-r1-zero project.
"""

import importlib.util
import os

try:
    from .hendrycks_math_grader import boxed_reward_fn as _boxed_reward_fn
except ImportError:
    _grader_path = os.path.join(os.path.dirname(__file__), "hendrycks_math_grader.py")
    _spec = importlib.util.spec_from_file_location("hendrycks_math_grader", _grader_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Failed to load hendrycks_math_grader from {_grader_path}")
    _grader_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_grader_module)
    _boxed_reward_fn = _grader_module.boxed_reward_fn


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    is_reflection = bool(extra_info.get("is_reflection", False))

    try:
        grading_info, correctness_score = _boxed_reward_fn(
            model_response=solution_str,
            gt_answer=ground_truth,
            fast=True,
        )
        formatted = grading_info.get("formatted", False)
    except Exception as e:
        print(f"Warning: MATH grading failed: {e}")
        correctness_score = 0.0
        formatted = False

    final_score = float(correctness_score)

    result = {
        "score": final_score,
        "acc": float(correctness_score > 0.0),
        "answer_present": formatted,
        "exact_match": correctness_score > 0.0,
    }

    if not is_reflection:
        result["score_base"] = final_score
    else:
        result["score_reflection"] = final_score

    return result
