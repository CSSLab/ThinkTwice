"""
MATH Dataset Reward Function for VERL
Wrapper around the high-recall math grader from understand-r1-zero project.

This module provides reward computation for the HENDRYCKS MATH dataset,
handling LaTeX boxed answers and symbolic mathematical expressions.
"""

import importlib.util
import os

# Import the canonical math grader.
#
# Notes:
# - When this reward fn is loaded via `load_module(file_path)` (Ray), there is no parent package, so relative imports
#   fail. We explicitly load the sibling module from its file path in that case.
# - Avoid mutating `sys.path`, since it can accidentally shadow third-party packages (e.g., `math_verify`).
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
    """
    Compute reward for a single MATH dataset response.

    This function extracts answers from LaTeX \\boxed{} commands and grades them
    using a high-recall grading system that handles symbolic math, fractions,
    expressions, and various LaTeX formats.

    This follows VERL's standard reward function interface (same as GSM8K).

    Args:
        solution_str: The model's response text
        ground_truth: The ground truth answer string
        data_source: Optional data source identifier (unused, for VERL compatibility)
        extra_info: Optional extra info dict, may contain 'is_reflection' flag
        **kwargs: Additional arguments (for VERL compatibility)

    Returns:
        dict with keys:
            - "score": float, 0.0 for wrong/unformatted, 1.0 for correct
            - "answer_present": bool, whether \\boxed{} was found
            - "exact_match": bool, whether answer matches ground truth
            - "score_base" or "score_reflection": float, same as score (for metric tracking)
    """
    extra_info = extra_info or {}

    # Use the canonical high-recall grading function
    try:
        grading_info, score = _boxed_reward_fn(
            model_response=solution_str,
            gt_answer=ground_truth,
            fast=True,  # Use fast mode (string + SymPy, no math_verify timeout)
        )

        # Extract whether answer was formatted
        formatted = grading_info.get("formatted", False)

    except Exception as e:
        # If grading fails, give zero reward
        print(f"Warning: MATH grading failed: {e}")
        score = 0.0
        formatted = False

    # Check if this is a reflection step
    is_reflection = bool(extra_info.get("is_reflection", False))

    # Build result dict (matches GSM8K format for metric compatibility)
    result = {
        "score": float(score),
        "answer_present": formatted,  # Whether \\boxed{} was found
        "exact_match": score > 0.0,  # Whether grading passed
    }

    # Add score tracking for base vs reflection
    if not is_reflection:
        result["score_base"] = float(score)
    else:
        result["score_reflection"] = float(score)

    return result
